# MLIR-VTA

基于 MLIR 的 VTA 加速器编译器，目标是逐步替换 `standalone-vta` 的 Python 两阶段编译器。
本仓库与上游 `standalone-vta/`（VTA ISA、功能/周期仿真器、Python 参考编译器）平级，只负责 **MLIR 侧的编译器**。

> 文档索引见 [`docs/README.md`](docs/README.md)。阶段一规格见 [`docs/specs/phase1-gemm-design.md`](docs/specs/phase1-gemm-design.md)；
> 阶段二 linalg 入口见 [`docs/specs/phase2-linalg-entry-design.md`](docs/specs/phase2-linalg-entry-design.md)；
> 阶段三通用 GEMM 见 [`docs/specs/phase3-generalized-gemm-design.md`](docs/specs/phase3-generalized-gemm-design.md)。

## 当前状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| 阶段一 | ✅ 完成 | 手写 `vtaisa.*` / `vta.gemm` → 二进制，与 Python 编译器 **字节级一致**，FSIM 验证 |
| 阶段二 | ✅ 完成（16×16 单块） | `linalg.matmul`(tensor) → tile/bufferize → `vta.gemm` → lower → translate → FSIM，结果矩阵与阶段一一致 |
| 阶段三 | ✅ 完成（通用 GEMM·任意 16 倍数维 + 多层 + overfit strategy-1/2/3/4） | `vta.gemm` 放宽到 16 倍数维度；通用化 `lower-vta-gemm`（块调度 + 多块数据 + 逐块 STORE + 依赖位 + **页对齐地址分配**）；方阵 32×32、矩形 32×48×16、3×3 方阵 48×48×48 端到端字节级 + FSIM。**多层编译**：`-vta-dram-allocation`（跨层 base 递增）+ 逐层独立工件，两层 16×16 共 15 文件字节级对齐上游。**Overfit strategy-1/2/3/4**：`vta.gemm {strategy=N}` 属性，四种溢出调度策略全部字节级验收（S1:16×2064×16, S2:192×16×192, S3:2064×16×16, S4:16×16×2048） |

| 阶段四 | 🔄 增量 A–C/E ✅；**F→G 进行中** | Python 桥接 + VTA 算子已验收；**下一步 F（CPU 算子）→ G（LeNet-5 整网）**；增量 D（onnx-mlir）暂缓（本地未装 onnx-mlir） |

**阶段四推进顺序（2026-06）：** F（QLinearAdd / Concat + `fsim_nn` 混合调度）→ G（LeNet-5 整网 `final_output.bin` vs `reference.bin`）→ D（待 onnx-mlir 环境就绪后替换 Python 桥接）。

**阶段三增量**（均已落地）：`--vta-semaphore-derive`、ALU lowering、`fsim_nn` 2×16×16 GEMM 串联（256/256 一致）。

## Dialect 与操作

编译器定义两个 dialect，分别对应高层算子与 VTA ISA 指令。

### 高层 `vta` dialect

| 操作 | 语法 | 说明 |
|------|------|------|
| `vta.gemm` | `vta.gemm ins(%lhs, %rhs : memref<MxKxi32>, memref<KxNxi32>) outs(%acc : memref<MxNxi32>)` | 块 GEMM：`acc += lhs * rhs`（原地累加在 `%acc` 上）。**无显式 `m,n,k` 属性**，块维 `Mb/Kb/Nb` 由 operand 形状推导；ODS verifier 要求各维为 **16 的倍数** 且 `lhs:MxK / rhs:KxN / acc:MxN` 形状匹配（阶段三放宽，阶段二仅认 16×16）。`lower-vta-gemm` 据形状做块调度生成指令；DRAM 缓冲基址由 `computeGemmLayout` 按 4 KiB 页对齐计算（复刻上游 `dram_allocation`，随矩阵尺寸变化），每块偏移按块算。可选 `{name="L0"}` 属性标记多层 NAME（文件后缀）。可选 `{mul_constant=N}` 表示 MulConstant GEMM（对角 WGT × 标量，跳过常规 K 维累加语义）。 |
| `vta.alu` | `vta.alu ins(%acc : memref<Mx16xi32>) {alu_opcode, use_imm, imm}` | 逐行元素 ALU：对 ACC/OUT 缓冲的 M 行向量执行标量运算。`alu_opcode`：0=MIN, 1=MAX, 2=ADD, 3=SHR, 4=MUL；`use_imm=true` 时 `imm` 为立即数（如 ADD_IMM 的 shift bias）。verifier 要求 acc 为 `[M, 16]` 的 2D memref（列必须是 16）。`lower-vta-alu` 按行发射 UOP 表 + 单 ALU 指令 + 逐行 STORE；DRAM 布局 ACC/OUT/UOP 页对齐（无 INP/WGT）。可选 `{name="L0"}` 属性。 |
| `vta.maxpool` | `vta.maxpool ins(%acc : memref<Mx16xi32>) {name?}` | 2×2 MaxPool（增量 E：36×16 黄金模式）。`lower-vta-maxpool` 发射四窗 vector-MAX + 部分 STORE。 |

**operand → 缓冲角色约定**（与数据发射器 / `memory_addresses.csv` 一致）：

| operand | 角色 | 数据文件 | `dram_base` |
|---------|------|----------|-------------|
| `%lhs` (ins[0]) | INP | `input.bin` | 64 |
| `%rhs` (ins[1]) | WGT | `weight.bin`（发射时转置写出） | 8 |
| `%acc` (outs[0]) | ACC / OUT | `accumulator.bin` / `out_init.bin` / 结果 @256 | 192 / 256 |

### 低层 `vtaisa` dialect

与 VTA 128-bit 宏指令 / 32-bit UOP 一一对应。所有 op 均为 **零结果**（side-effect 指令），字段通过属性表达。

#### 片上缓冲枚举 `BufferId`

| 值 | 名称 | 含义 |
|----|------|------|
| 0 | UOP | 微操作表缓冲 |
| 1 | WGT | 权重 |
| 2 | INP | 输入激活 |
| 3 | ACC | 累加器 |
| 4 | OUT | 输出 |

#### 指令 op 一览

| 操作 | 主要属性 | 说明 |
|------|----------|------|
| `vtaisa.uop_table` | `dst`, `src`, `wgt` : `i64` 数组 | 微操作表条目（每条 UOP 的 dst/src/wgt 索引） |
| `vtaisa.load` | `buffer_id`, `sram_base`, `dram_base`, `y_size`, `x_size`, `x_stride`, `y_pad_*`, `x_pad_*`, `pop_prev/next`, `push_prev/next` | DRAM → SRAM 加载 |
| `vtaisa.store` | `buffer_id`, `sram_base`, `dram_base`, `y_size`, `x_size`, `x_stride`, `pop_prev/next`, `push_prev/next` | SRAM → DRAM 写回 |
| `vtaisa.gemm_insn` | `uop_bgn`, `uop_end`, `loop_out`, `loop_in`, `dst/src/wgt_factor_*`, `reset`, `pop_prev/next`, `push_prev/next` | GEMM 微操作循环 |
| `vtaisa.alu_insn` | 同 `gemm_insn` 因子字段 + `alu_opcode`, `use_imm`, `imm` | ALU 微操作循环（由 `lower-vta-alu` 生成） |
| `vtaisa.finish` | （无） | 程序结束标记 |

**GEMM 指令序列**（由 `-lower-vta-gemm` 生成，CASE 1 单 step）：`uop_table`(1 reset + G 条 GeMM uop) + `LOAD UOP(reset)` + `GEMM reset` + 合并 `LOAD INP/WGT/ACC`(`y_size=块数`) + `LOAD UOP(gemm)` + 主 `GEMM`(`uop[0,G)`) + **逐块 `STORE`**(`nbC` 条) + 2 条终止 NOP + `finish`，共 `(10 + nbC)` 条 ISA 指令。16×16（`Mb=Kb=Nb=1`）退化为 **11 条** 指令 + 2 UOP，与第一阶段字节级一致；32×32（2×2 块）为 14 条指令 + 9 UOP。示例见 `test/Target/gemm16x16.mlir`、`test/Target/matmul_tensor_32x32.mlir`。

**ALU 指令序列**（由 `-lower-vta-alu` 生成）：`uop_table`(1 reset + M 条 ALU uop，每行 dst=i,src=0,wgt=0) + `LOAD UOP(reset)` + `GEMM reset`(push_prev=1) + `LOAD ACC`(M/16 块) + `LOAD UOP(alu)` + 主 `ALU`(uop[0,M),push_next=1) + **逐行 `STORE`**(`M` 条) + 2 条终止 NOP + `finish`，共 `(8 + M)` 条。ADD_IMM 16×16（M=16 行）：**24 条** 指令 + 17 UOP，字节级对齐上游 `alu_16x16` 黄金。DRAM 布局（无 INP/WGT）：ACC=0x40, OUT=0x80, UOP=0xc00（logical）。

### 编译 Pass（`vta-opt`）

| Pass | 参数 | 作用 |
|------|------|------|
| `ConvertLinalgToVTA` | `--convert-linalg-to-vta` | 已 bufferize 的 `linalg.matmul`（memref 语义，任意 16 倍数维度）→ `vta.gemm`；若仍是 tensor 语义则报错 |
| `VTADramAllocation` | `--vta-dram-allocation` | 多层：按出现顺序遍历各 `vta.gemm` / `vta.alu` / `vta.maxpool`，跨层页对齐 base 递增（复刻上游 `updated_base_address`），把逐层物理/逻辑基址写入模块属性 `vta.layers`，供 lowering/发射器逐层取用 |
| `LowerVTAGemm` | `--lower-vta-gemm` | `vta.gemm`(N×N) → `vtaisa.*` 指令序列；有 `vta.layers` 时按层名取逻辑基址，否则单层 `computeGemmLayout`。CASE 1（无溢出）：单步；Strategy-1/2/3/4（溢出，由 `{strategy=N}` 属性选择）：多步 LOAD/GEMM/STORE；`{mul_constant=N}` 走 MulConstant 专用路径 |
| `LowerVTAAlu` | `--lower-vta-alu` | `vta.alu`(M×16 memref) → `vtaisa.*` 指令序列；按行发射 ALU UOP 表 + `GEMM reset` + `LOAD ACC` + `LOAD UOP(alu)` + 主 `ALU insn` + M 条 `STORE` + NOP + `finish`。DRAM 布局通过 `computeAluLayout` 页对齐（顺序 ACC/OUT/UOP/INSN，无 INP/WGT）。ADD_IMM 16×16 字节级对齐上游黄金（24 insns + 17 uops）。 |
| `LowerVTAMaxPool` | `--lower-vta-maxpool` | `vta.maxpool`(36×16 模式) → 四窗 vector-MAX + 部分 STORE；字节级对齐 `maxpool_36x16` 黄金 |
| `VTASemaphoreDerive` | `--vta-semaphore-derive` | 在 `vtaisa.*` 序列上运行 4 计数器信号量算法（CMP↔LD↔ST），重新推导所有 `pop_prev/pop_next/push_prev/push_next` 依赖位，取代硬编码；对所有 GEMM/ALU 策略产生与黄金字节级一致的结果 |

`vta-opt` 同时注册全部上游 dialect/pass（`-linalg-tile`、各 `*-bufferize`、`-canonicalize` 等），可直接跑完整中端管道。

## 前置条件

| 依赖 | 说明 |
|------|------|
| LLVM / MLIR **13.0.0** | 本机已装于 `/usr/local/llvm`（含 `MLIRConfig.cmake`、`mlir-tblgen`） |
| 编译器 | `clang++` / `clang`（与 LLVM 13 ABI 匹配；不要用 g++） |
| 构建工具 | CMake ≥ 3.20 + `make`（无 `ninja`，用 `Unix Makefiles`） |
| Python（仅参考/验证） | `python3` + `numpy`，用于跑 `standalone-vta` 生成黄金参考与 FSIM |

> 构建目录建议放在仓库外（如 `~/mlir-vta-build`），避免占用 `/mnt/c` 空间。

## 构建

```bash
cmake -S . -B ~/mlir-vta-build -G "Unix Makefiles" \
  -DMLIR_DIR=/usr/local/llvm/lib/cmake/mlir \
  -DLLVM_DIR=/usr/local/llvm/lib/cmake/llvm \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
  -DCMAKE_BUILD_TYPE=Release
cmake --build ~/mlir-vta-build -j"$(nproc)" --target vta-opt --target vta-translate
```

产物：`~/mlir-vta-build/bin/vta-opt`、`~/mlir-vta-build/bin/vta-translate`。

## 使用

### 1) 阶段一：高层 `vta.gemm` → ISA → 二进制

```bash
# 展开 vta.gemm 为 vtaisa 指令序列（含 uop_table + finish）
~/mlir-vta-build/bin/vta-opt -lower-vta-gemm test/Target/lower_gemm.mlir -o /tmp/lowered.mlir
# 发射二进制（输出目录需预先存在）
mkdir -p /tmp/out
~/mlir-vta-build/bin/vta-translate /tmp/lowered.mlir -o /tmp/out
ls /tmp/out   # instructions.bin  uop.bin
```

> `vta-translate` 不会自动创建 `-o` 目录，运行前请先 `mkdir -p`。

### 2) 阶段一：直接发射手写低层 MLIR + 数据/CSV

```bash
mkdir -p /tmp/out
~/mlir-vta-build/bin/vta-translate test/Target/gemm16x16.mlir -o /tmp/out \
  --emit-data --input test/golden/input_16x16.bin --weight test/golden/weight_16x16.bin
```

### 3) 阶段二：`linalg.matmul`(16×16 tensor) 完整管道

```bash
~/mlir-vta-build/bin/vta-opt test/Target/matmul_tensor.mlir \
  --linalg-tile="linalg-tile-sizes=16,16,16" \
  --linalg-bufferize --tensor-bufferize --tensor-constant-bufferize \
  --func-bufferize --finalizing-bufferize \
  --convert-linalg-to-vta --canonicalize --lower-vta-gemm \
  -o /tmp/lowered.mlir

mkdir -p /tmp/out
~/mlir-vta-build/bin/vta-translate /tmp/lowered.mlir -o /tmp/out \
  --emit-data --input test/golden/input_16x16.bin --weight test/golden/weight_16x16.bin
```

> 入口 IR 用 `constant 0 : i32`（非 `arith.constant`），因本机 `vta-opt` 未注册 `arith` dialect。

### 4) 阶段三：`linalg.matmul`(32×32 tensor) 多块管道

通用维度（16 的倍数）GEMM。**不用 `--linalg-tile`**——块调度在 `--lower-vta-gemm` 内部完成。
数据发射用 `--rows/--cols/--k`（元素维，默认 16）指定矩阵尺寸，按块主序写出（WGT 每块转置）。

```bash
# 入口 32×32 tensor → bufferize → vta.gemm(32×32) → 通用 lower → vtaisa
~/mlir-vta-build/bin/vta-opt test/Target/matmul_tensor_32x32.mlir \
  --linalg-bufferize --tensor-bufferize --tensor-constant-bufferize \
  --func-bufferize --finalizing-bufferize \
  --convert-linalg-to-vta --canonicalize --lower-vta-gemm \
  -o /tmp/lowered32.mlir

# 发射指令/UOP + 多块数据/CSV（C[32×32] = A[32×32] · B[32×32]）
mkdir -p /tmp/out32
~/mlir-vta-build/bin/vta-translate /tmp/lowered32.mlir -o /tmp/out32 \
  --emit-data --rows 32 --cols 32 --k 32 \
  --input test/golden/matmul_32x32/input_32x32.bin \
  --weight test/golden/matmul_32x32/weight_32x32.bin
```

> 32×32 → 2×2 块（CASE 1 无 overfit，单 step）：14 条指令 + 9 条 UOP + 4 条逐块 STORE。
> `--rows M --cols N --k K` 对应 `A[M×K]`、`B[K×N]`、`C[M×N]`，三者均须为 16 的倍数。
>
> **矩形 / 更多块同理**（块维由形状推导）。入口与黄金已备：
> `test/Target/matmul_tensor_32x48x16.mlir`（`C[32×16]=A[32×48]·B[48×16]`，2×3×1 块）、
> `test/Target/matmul_tensor_48x48x48.mlir`（3×3 块）。注意 **DRAM 基址随尺寸变化**：
> `32×48×16` 的 INP 占 1.5 页，把 WGT/ACC/OUT/UOP 逻辑基址顶到 12/256/320/6144（不再是
> 16×16/32×32 的 8/192/256/5120），`vta-translate` 的 `memory_addresses.csv` 会自动跟随。

### 4b) 多层编译（单 func 内多个 `vta.gemm`）

每个 `vta.gemm` 带 `{name="L0"}` 即一层；`--vta-dram-allocation` 跨层递增 DRAM base，发射器按 `finish` 切分逐层产 `instructions<NAME>.bin`/`uop<NAME>.bin` 等，并写多行 `layers_name.csv`。数据发射用可重复的 `--layer NAME=M,K,N,inputpath,weightpath`。

```bash
# 入口：单 func 内 vta.gemm{name=L0} + vta.gemm{name=L1}
~/mlir-vta-build/bin/vta-opt test/Target/matmul_2layer_16x16.mlir \
  --vta-dram-allocation --lower-vta-gemm -o /tmp/lowered2.mlir

# 逐层发射指令/UOP/数据/CSV
mkdir -p /tmp/out2
~/mlir-vta-build/bin/vta-translate /tmp/lowered2.mlir -o /tmp/out2 \
  --layer L0=16,16,16,test/golden/matmul_2layer_16x16/inputL0.bin,test/golden/matmul_2layer_16x16/weightL0.bin \
  --layer L1=16,16,16,test/golden/matmul_2layer_16x16/inputL1.bin,test/golden/matmul_2layer_16x16/weightL1.bin

# 一键字节级验收（15 文件对齐上游黄金）
bash scripts/verify_2layer.sh
```

> 上游 `fsim_single_layer` 无法执行累积多层地址（见上文限制说明），故多层以**字节级对齐参考编译器**为正确性 gate。

### 5) 字节级回归（对比黄金参考）

```bash
# 阶段一/二（16×16）
cmp /tmp/out/instructions.bin   test/golden/instructions.bin              && echo "16x16 INSN OK"
cmp /tmp/out/uop.bin            test/golden/uop.bin                       && echo "16x16 UOP OK"
# 阶段三（32×32：指令/UOP/数据 bin 全字节级一致）
cmp /tmp/out32/instructions.bin test/golden/matmul_32x32/instructions.bin && echo "32x32 INSN OK"
cmp /tmp/out32/uop.bin          test/golden/matmul_32x32/uop.bin          && echo "32x32 UOP OK"
cmp /tmp/out32/input.bin        test/golden/matmul_32x32/input.bin        && echo "32x32 INP OK"
cmp /tmp/out32/weight.bin       test/golden/matmul_32x32/weight.bin       && echo "32x32 WGT OK"

# 矩形 / 更多块：8 文件（含 CSV）字节级 + FSIM 结果一站式验收
scripts/run_fsim_gemm.sh 32 48 16   # C[32×16]=A[32×48]·B[48×16]，对齐 test/golden/matmul_32x48x16/
scripts/run_fsim_gemm.sh 48 48 48   # 3×3 块，对齐 test/golden/matmul_48x48x48/
```

### 6) 重新生成黄金参考（固定随机种子）

```bash
scripts/make_golden.sh           # 调用 standalone-vta 的 Python 编译器，刷新 test/golden/（16×16）
scripts/make_golden_gemm.sh 32 48 16  # 任意 16 倍数 M K N → test/golden/matmul_MxKxN/（8 文件 + FSIM 结果）
```

> 32×32 黄金（`test/golden/matmul_32x32/`）由上游 `main_vta_compiler.py` 跑
> `standalone-vta/examples/vta_ir/matmul_32x32.json` 生成，流程见
> [`docs/plans/spike-generalized-gemm-notes.md`](docs/plans/spike-generalized-gemm-notes.md)；
> 矩形/更多块黄金用 `scripts/make_golden_gemm.sh <M> <K> <N>` 生成。

### 8) ALU lowering（`vta.alu` → vtaisa 指令序列）

```bash
# 手写 vta.alu IR（ADD_IMM 16×16，M=16 行）
cat > /tmp/alu16.mlir << 'EOF'
func @main(%acc: memref<16x16xi32>) {
  vta.alu ins(%acc : memref<16x16xi32>) {alu_opcode = 2 : i64, use_imm = true, imm = 3 : i64}
  return
}
EOF

# lower → vtaisa 序列
~/mlir-vta-build/bin/vta-opt --lower-vta-alu /tmp/alu16.mlir -o /tmp/alu16_low.mlir

# 发射二进制
mkdir -p /tmp/alu_out
~/mlir-vta-build/bin/vta-translate /tmp/alu16_low.mlir -o /tmp/alu_out

# 字节级验收（24 insns + 17 uops）
cmp /tmp/alu_out/instructions.bin test/golden/alu_add_imm_16x16/instructions.bin && echo "ALU INSN OK"
cmp /tmp/alu_out/uop.bin          test/golden/alu_add_imm_16x16/uop.bin          && echo "ALU UOP OK"
```

> `alu_opcode`：0=MIN, 1=MAX, 2=ADD, 3=SHR, 4=MUL。`use_imm=true` 使用立即数 `imm`（ADD_IMM 的 shift bias）。
> DRAM 布局（页对齐，无 INP/WGT）：ACC=0x40, OUT=0x80, UOP=0xc00（logical）。

### 9) FSIM 端到端验证

```bash
scripts/run_fsim.sh              # 阶段一：gemm16x16.mlir → FSIM
scripts/run_fsim_linalg.sh       # 阶段二：matmul_tensor.mlir 完整管道 → FSIM + 结果矩阵自校验
scripts/run_fsim_linalg_32x32.sh # 阶段三：matmul_tensor_32x32.mlir 多块管道 → FSIM + 4 块结果矩阵自校验
scripts/run_fsim_gemm.sh M K N   # 阶段三通用：任意 16 倍数维 → 完整管道 → FSIM + 结果矩阵对齐黄金
scripts/run_fsim_overfit.sh      # 阶段三 overfit：matmul_tensor_overfit.mlir (16×2064×16, strategy-1) → FSIM
scripts/run_onnx_qconv.sh        # 阶段四增量 A：qlinearconv_debug.onnx → MLIR → 字节级 + FSIM
scripts/run_onnx_qconv_colpad.sh # 阶段四增量 B：col-pad + expand_bias QLinearConv
scripts/run_onnx_two_qconv.sh  # 阶段四增量 C：two_qlinearconv → fsim_nn
scripts/run_onnx_relu.sh         # 阶段四增量 E：ReLU 16×16 MAX_IMM → vta.alu → 字节级 + FSIM
scripts/run_onnx_maxpool.sh      # 阶段四增量 E：MaxPool 36×16 → vta.maxpool → 字节级 + FSIM
scripts/run_onnx_qlinearmul.sh   # 阶段四增量 E：QLinearMul MulConstant → vta.gemm → 字节级 + FSIM
scripts/run_onnx_conv_relu.sh    # 阶段四增量 E：Conv+Relu 两层 → fsim_nn + reference.bin 数值对比
scripts/run_onnx_qadd_two_mul.sh # 阶段四增量 F-a：2×Mul + QLinearAdd → fsim_nn + 数值 golden
scripts/run_onnx_qconcat.sh      # 阶段四增量 F-b：2×Conv + QLinearConcat → fsim_nn + 数值 golden
scripts/run_onnx_lenet5.sh       # 阶段四增量 G：lenet5_mini 2×(Conv+Relu) → fsim_nn + 数值 golden
```

## 工程结构

```
mlir-vta/
├── CMakeLists.txt
├── include/mlir-vta/
│   ├── Dialect/VTA/        # 高层: VTADialect.td / VTAOps.td / VTAPasses.h
│   ├── Dialect/VTAISA/     # 低层: VTAISADialect.td / VTAISAOps.td / VTAISAEnums.td
│   └── Target/             # VTABinaryEmitter.h / VTADataEmitter.h
├── lib/
│   ├── Dialect/VTA/        # 高层 dialect + op 实现
│   ├── Dialect/VTAISA/     # 低层 dialect + op 实现
│   ├── Target/             # 二进制 + 数据/CSV 发射器
│   └── Transforms/         # LowerVTAGemm / ConvertLinalgToVTA
├── tools/
│   ├── vta-opt/            # 注册 vta+vtaisa dialect + pass 的 mlir-opt 变体
│   └── vta-translate/      # MLIR → 二进制/数据/CSV
├── test/
│   ├── Dialect/            # round-trip（vta.gemm ins/outs 格式，含 32×32）
│   ├── Target/             # gemm16x16.mlir / lower_gemm.mlir / matmul_tensor*.mlir
│   └── golden/             # 字节级黄金 fixtures + fsim_result_16x16.txt + matmul_32x32/
├── scripts/                # make_golden.sh / run_fsim*.sh / run_fsim_linalg*.sh
└── docs/                   # README.md、DESIGN_cn.md、plans/、specs/
```

## 测试约定

本环境未安装 `FileCheck`，故 `.mlir` 顶部 `// RUN: ... FileCheck` 行仅作文档注释；
round-trip 校验用 `vta-opt %s | vta-opt` + `grep`，字节级校验用 `cmp`（见上）。
