# 阶段三增量：多层 DRAM 地址分配（忠实多层编译）

> 目标：把单层的 `computeGemmLayout` 推广为**跨层游标**分配，交付独立 `-vta-dram-allocation`
> pass，按层产出独立工件（`instructions<NAME>.bin` 等）+ 多行 `layers_name.csv`，
> 用 `fsim_single_layer` 逐层验收。**不**做真·层间数据串联（属 `fsim_nn` 子系统，明确推迟）。

## 上游事实（已用两层黄金验证）
- 多层 = 多个独立单算子层；每层各自 `instructions<NAME>.bin`/`uop<NAME>.bin`/`input<NAME>.bin`…，**不拼接**，各有自己的 `FINISH`。
- 一个页对齐分配器带**跨层游标**连续往上排；逻辑地址 = 物理/divisor（**全局递增**）。
  - L0：INP=64/WGT=8/ACC=192/OUT=256/UOP=5120，物理 0x1000→末 0x60af。
  - L1：INP=448/WGT=32/ACC=576/OUT=640/UOP=11264，物理 0x7000→末 0xc0af（接 L0 末地址页对齐）。
- `layers_name.csv`：N 行 `<i>,<NAME>,<hex 末物理地址>`。
- `fsim_single_layer` 逐层独立执行（各读自己 `input<NAME>.bin`），层间不串数据。

## MLIR 表示
单 `func` 内多个 `vta.gemm`（按出现顺序为层）。`vta.gemm` 增可选 `name`（StrAttr，默认 `""`）= 层 NAME = 文件后缀。单层旧测试无 `name` → `instructions.bin`（无后缀），保持回归字节级一致。

## 实现步骤（每步带字节级 gate）
1. **`VTAGemmLayout.h`**：加 `computeGemmLayoutAt(Mb,Kb,Nb,baseCursor)`；`computeGemmLayout(...)=...At(...,0)`。游标 = 上层 `lastPhys`。
2. **`VTAOps.td`**：`vta.gemm` 加可选 `StrAttr:$name`（默认 `""`）。
3. **新 pass `-vta-dram-allocation`**（`lib/Transforms/VTADramAllocation.cpp`）：按序遍历 `vta.gemm`，线性穿引游标，构造模块属性 `vta.layers`（`ArrayAttr<DictionaryAttr>`，每项含 name/M/K/N + INP/WGT/ACC/OUT/UOP/INSN 的 phys+logical + lastPhys）。注册到 `registerVTAPasses`。
4. **`LowerVTAGemm`**：第 i 个 gemm 若模块有 `vta.layers` → 用第 i 项的逻辑基址；否则 `computeGemmLayout(...,0)`（单层回退，字节级不变）。其余发射逻辑不动（每层以 finish 结尾）。
5. **`VTABinaryEmitter`**：模块有 `vta.layers` → 多层模式：按 `finish` 切分 op 组，逐层写 `instructions<NAME>.bin`+`uop<NAME>.bin`；写 `layers_name.csv`（N 行）+ `memory_addresses<NAME>.csv`（逐层）。无 `vta.layers` → 单文件（现状）。
6. **`VTADataEmitter` + `vta-translate` CLI**：加可重复 `--layer NAME=M,K,N,inputpath,weightpath`。多层模式逐层产 `input<NAME>.bin`/`weight<NAME>.bin`/`accumulator<NAME>.bin`/`metadata<NAME>.csv`；CSV（memory_addresses/layers_name）归 binary emitter。单层模式（无 `--layer`）维持现状。

## 验收
- 字节级：`instructionsL0/L1.bin`、`uopL0/L1.bin`、`layers_name.csv`、`memory_addressesL0/L1.csv`、`metadataL0/L1.csv`、`inputL0/L1.bin`、`weightL0/L1.bin`、`accumulatorL0/L1.bin` 全对齐 `test/golden/matmul_2layer_16x16/`。
- FSIM：`fsim_single_layer` 跑两层，结果矩阵对齐黄金。
- 回归：16×16 / 32×32 / 32×48×16 / 48×48×48 单层字节级 + FSIM 不变。

## 实施结果（已完成）
- ✅ `-vta-dram-allocation` pass：跨层游标分配，写模块属性 `vta.layers`。
- ✅ `vta.gemm` 加可选 `name`；lowering 按 `vta.layers` 用各层逻辑基址；binary emitter 按 `finish` 切分逐层产 `instructions<NAME>.bin`/`uop<NAME>.bin` + `layers_name.csv` + `memory_addresses<NAME>.csv`；data emitter `--layer` 逐层产数据/metadata。
- ✅ **15/15 文件字节级对齐**上游两层黄金 `test/golden/matmul_2layer_16x16/`（`scripts/verify_2layer.sh`）。
- ✅ 单层 16/32/矩形/3×3 字节级 + FSIM 回归无破坏。

## FSIM 限制（关键发现，已验证）
`fsim_single_layer` 逐层 `VTAMemAlloc`/`VTAMemFree`，分配器**层间回收复用低地址**，无法执行上游多层编译的**累积地址**（L1 INP=0x7000+）：运行到 L1 的 UOP（phys 0xB000）时页表只有 6 页 → 崩溃。**上游黄金原件同样崩溃**，证明这是上游 `fsim_single_layer` 对累积多层的固有限制，非本实现问题。
- 因此多层验收的"正确性 gate"= **字节级对齐参考编译器输出**（参考编译器输出即正确性基准）。
- 各层计算正确性已被单层 16×16 FSIM 覆盖：L0 指令流与标准单层 16×16 **逐字节相同**，L1 仅地址重定位、结构相同。
- 真·层间执行/数据串联走 `fsim_nn`（见下）。

## 明确推迟
- 真·层间数据串联与执行（`fsim_nn` + `dependency.csv` + im2row，conv 子系统）。
- 独立 pass 注解物理地址到具体 op（当前用模块级 `vta.layers`）。
