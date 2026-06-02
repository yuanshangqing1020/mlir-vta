# MLIR-VTA 第二阶段设计 · `linalg.matmul` 入口（16×16 单块）

> **文档定位：** 本文是第二阶段的**设计规格（spec）**，经头脑风暴确认后写出，作为后续实现计划（writing-plans）的输入。
> - 第一阶段设计规格见 [`phase1-gemm-design.md`](phase1-gemm-design.md)；实施计划与黄金 hex 全文见 [`../plans/phase1-gemm.md`](../plans/phase1-gemm.md)；
> - 架构总览见 [`../DESIGN_cn.md`](../DESIGN_cn.md)。
> - 若代码与本文不一致，**以源码为准**，并同步更新本文。

---

## 1. 目标（一句话）

在 16×16 单块这个固定用例上，立起一条**真实的 MLIR 中端链路**：
`linalg.matmul`(tensor) → `--linalg-tile` → bufferization → `vta.gemm`(memref operand) → 复用第一阶段单块 lowering → `vta-translate` → FSIM，
使 FSIM 结果矩阵与第一阶段（即 Python/numpy 参考）在同一 fixture 下**逐元素一致**。

## 2. 范围决策（头脑风暴结论）

| 维度 | 决策 |
|------|------|
| **范围切法** | 先打通 linalg 入口（16×16 单块），**不做**多块/通用维度。把 `linalg → vta → bin` 新链路锁死，通用化单独留一阶段。 |
| **基础设施** | A3 完整管道：tensor → tiling → bufferization → `vta.gemm`，路管从头走一遍（16×16 下 tiling/bufferization 部分是 no-op，但结构真实存在）。 |
| **验收标准** | `fsim_only`：仅要求 FSIM 结果矩阵 == 第一阶段同 fixture 的 FSIM 结果（numpy 参考）。**不**强绑指令字节级与旧 golden 一致。 |
| **`vta.gemm`** | 扩展为承载 memref operand（`ins %A,%B outs %C`），使 bufferization 真正作用于其上。 |
| **管道顺序** | 方案 2：先在 tensor 上 tile，再 bufferize，再 convert（更现代；LLVM 13 风险用 spike 兜底，必要时回退方案 1）。 |

## 3. 架构与数据流

```
func @main(tensor %A,%B)            ← 入口（tensor 级 linalg.matmul + fill 初始化 ACC=0）
  │  --linalg-tile=16,16,16         （tensor 上单块；产生 extract_slice/insert_slice）
  │  bufferization 栈               （linalg/tensor/func/finalizing-bufferize → memref）
  │  --convert-linalg-to-vta        （新 pass：linalg.matmul → vta.gemm(memref operand)）
  │  --canonicalize                 （折叠单次循环、清理死 slice）
  │  --lower-vta-gemm               （复用：单块 11 条 vtaisa + 2 UOP，硬编码黄金地址/依赖位）
  ▼
vtaisa.* 指令流  ── vta-translate ──► instructions.bin / uop.bin
                  vta-translate --emit-data ──► input/weight/accumulator/out_init.bin + 3 CSV
                                                       │
                                                       ▼  run_fsim.sh
                                                     FSIM 结果矩阵
```

## 4. 入口 IR（`test/Target/matmul_tensor.mlir`）

```mlir
func @main(%A: tensor<16x16xi32>, %B: tensor<16x16xi32>) -> tensor<16x16xi32> {
  %c0 = arith.constant 0 : i32
  %init = linalg.init_tensor [16, 16] : tensor<16x16xi32>   // LLVM 13：linalg.init_tensor（非 tensor.empty）
  %acc = linalg.fill(%c0, %init) : i32, tensor<16x16xi32> -> tensor<16x16xi32>
  %out = linalg.matmul
           ins(%A, %B : tensor<16x16xi32>, tensor<16x16xi32>)
           outs(%acc : tensor<16x16xi32>) -> tensor<16x16xi32>
  return %out : tensor<16x16xi32>
}
```

> 入口 IR 的实际数值不在 IR 内提供；输入/权重数据仍来自外部 `.bin`（数据发射器，固定 DRAM 地址）。operand 仅作类型/形状载体与「buffer 角色」约定。

## 5. `vta.gemm` op 扩展

- 由「纯 `m,n,k` 属性」改为带 memref operand：`vta.gemm ins(%A, %B) outs(%C)`。
- `m,n,k` 从 memref 形状推出（不再要求显式属性）。
- **verifier**：本阶段断言三 operand 均为 `memref<16x16xi32>`（非此形状报错）。
- memref operand 在本阶段是**信息性的**（供未来地址分配 pass 使用）；`lower-vta-gemm` 仍用第一阶段硬编码的 `dram_base`/依赖位生成 11 条指令，不读 operand 地址。
- `assemblyFormat` 带 operand 与类型列表，保证 round-trip。

## 6. 新 pass：`--convert-linalg-to-vta`

- 文件：`lib/Transforms/ConvertLinalgToVTA.cpp`，声明并入 `VTAPasses.h`，由 `vta-opt` 注册。
- 形态：匹配 bufferize 后的 `linalg::MatmulOp`（memref 语义）。
- 校验：lhs/rhs/out 均 `memref<16x16xi32>`，否则 `emitError` + `signalPassFailure`（本阶段不支持其它尺寸）。
- 动作：在原位创建 `vta.gemm ins(lhs, rhs) outs(out)`；擦除 `linalg.matmul`；将对应的 `linalg.fill`(0)（ACC 初始化）折进 `vta.gemm` 的 reset 语义（即丢弃 fill，因为单块 lowering 已含 reset+ACC=0）。
- operand → 缓冲角色约定（见 §7）由该 pass 通过 operand 顺序固定：`ins[0]=INP, ins[1]=WGT, outs[0]=ACC/OUT`。

## 7. operand → DRAM 缓冲映射（沿用第一阶段，不改数据布局）

| operand | 缓冲 | 数据文件 | dram_base | 备注 |
|---------|------|----------|-----------|------|
| `%A` (lhs) | INP | `input.bin` | 64 | 原样拷贝 |
| `%B` (rhs) | WGT | `weight.bin` | 8 | 数据发射器仍做转置写出 |
| `outs` 结果 | ACC / OUT | `accumulator.bin`/`out_init.bin`（全 0）/ OUT@256 | 192 / 256 | 复用 |

`VTADataEmitter`（数据 bin + 3 CSV）**原样复用，不动**。

## 8. 数值约定（重要边界）

本阶段**不重新推导**矩阵乘语义。理由：数据发射器与 `lower-vta-gemm` 完全复用第一阶段 ⇒ FSIM 结果与第一阶段**逐字节相同** ⇒ 第一阶段已验证 == Python 路径。
因此 `linalg.matmul` 在本阶段是「被识别的标记」；它的数学约定（`A·B` vs VTA 的 `INP·WGTᵀ` + 磁盘转置）与通用化的对应关系**记为文档说明 + 留给通用化阶段**，不在本阶段展开。

## 9. emitter/lowering 的遍历调整

`vta.gemm`(带 operand) 与降级后的 `vtaisa.*` 会嵌在 `func`（可能还有残留 `scf`）体内，而非 module 顶层。
- `lower-vta-gemm` 已用 `module.walk`（递归），保留。
- `VTABinaryEmitter` 改为**递归遍历**（`module.walk`，按 IR 文本/前序，对线性 func body 即指令顺序），以容忍 op 嵌在 `func`/`scf` 内；fail-loud 行为（未处理 vta/vtaisa op 报错）保持不变。

## 10. 验证策略（验收 = FSIM 结果一致）

| 层级 | 手段 |
|------|------|
| 逐 pass IR 检查 | `vta-opt <pass> %s \| grep -q '<期望片段>'`（环境无 FileCheck） |
| convert pass 单元 | bufferize 后 IR → `--convert-linalg-to-vta` → `grep -q 'vta.gemm'` 且无残留 `linalg.matmul` |
| 端到端（验收门） | `matmul_tensor.mlir` 跑完整管道 → `vta-translate` → `run_fsim.sh` → 结果矩阵 == 第一阶段同 fixture 的 FSIM 结果（numpy 参考） |
| 非阻断信号 | 对现有 16×16 golden `cmp`（lowering 未变，理应仍一致；**不作为验收门**） |

## 11. 风险与缓解

| 风险 | 缓解 |
|------|------|
| LLVM 13 tile-on-tensors + slice bufferization 不成熟（方案 2） | **实现第一步做 spike**：纯手跑 `mlir-opt` 验证 `--linalg-tile` → bufferization 产出可转换 IR；走不通则回退方案 1（先 bufferize 再在 memref 上 tile），并记录决策 |
| `linalg.init_tensor`/`fill` 的 bufferization 边角 | spike 一并验证；必要时显式加 `--tensor-constant-bufferize` |
| `vta.gemm` 嵌套在 `func`/`scf` 内 emitter 看不到 | §9 emitter 改递归遍历 |
| tile=full 产生单次 `scf.for` 残留 | `--canonicalize` 折叠；convert pass 对内层 `linalg.matmul` 的匹配不依赖外层循环 |

## 12. 明确不做（推迟到后续阶段）

- 多块 / 通用 `m,n,k` 的 tiling 与数据重排；
- 地址分配 pass（`dram_allocation` 复刻，本阶段保持硬编码）；
- 依赖信号量自动推导；
- ALU lowering；
- onnx-mlir 前端。

## 13. 交付物清单

- 扩展 `include/.../VTA/VTAOps.td`（`vta.gemm` 带 operand + verifier）+ `lib/Dialect/VTA/VTAOps.cpp`。
- 新 `lib/Transforms/ConvertLinalgToVTA.cpp` + `VTAPasses.h` 注册 + `vta-opt` 注册。
- `lib/Transforms/LowerVTAGemm.cpp` 适配带 operand 的 `vta.gemm`（仍生成黄金 11 条）。
- `lib/Target/VTABinaryEmitter.cpp` 改递归遍历。
- 入口 `test/Target/matmul_tensor.mlir` + 逐 pass `grep` 断言脚本。
- `scripts/run_fsim.sh` 复用 / 微调以接新入口。
- 更新 `DESIGN_cn.md`（§10.1 状态由「规划中」→「进行中/已落地」，记录数值约定边界与 spike 决策）。
