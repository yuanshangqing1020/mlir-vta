# MLIR-VTA 第三阶段设计 · 通用维度 GEMM（多块 · 先打通 32×32×32）

> **文档定位：** 本文是第三阶段的**设计规格（spec）**，经分析上游 Python 参考编译器后写出，作为实施计划（plan）的输入。
> - 第二阶段（linalg 入口·16×16 单块）见 [`phase2-linalg-entry-design.md`](phase2-linalg-entry-design.md)；
> - Task 0 spike（32×32 黄金解码）见 [`../plans/spike-generalized-gemm-notes.md`](../plans/spike-generalized-gemm-notes.md)；
> - 架构总览见 [`../DESIGN_cn.md`](../DESIGN_cn.md)。
> - 若代码与本文不一致，**以源码为准**，并同步更新本文。

---

## 1. 目标（一句话）

把 `vta.gemm` 从「仅 16×16 单块」扩展到**任意 16 的倍数维度**（先打通 **32×32×32**，即 2×2 块），
端到端链路 `linalg.matmul`(N×N tensor) → bufferize → `vta.gemm`(N×N memref) → **通用化 lowering** → `vta-translate`（多块数据/CSV）→ FSIM，
使 FSIM 结果矩阵与上游 Python 编译器同 fixture 的结果**逐元素一致**（验收门）；同时 `instructions.bin`/`uop.bin`/数据 bin 对上游黄金**字节级一致**（强信号）。

## 2. 范围决策

| 维度 | 决策 |
|------|------|
| **第一个目标** | **32×32×32**（各维 2 块，`nb_A=nb_B=nb_C=4`）。这是能强迫出真实多块 tiling + K 维 reduce + 多块地址 + 多 STORE 的**最小**用例。 |
| **是否用 MLIR linalg-tile** | **不用**。上游的"分块"是自定义块调度（单 step、单 GEMM 指令带 N 条 uop、合并 LOAD），与 MLIR 循环 tiling 产出的 scf 嵌套形态不符。改为让 `vta.gemm` 承载完整 N×N memref，由 lowering **内部**复刻上游块调度。 |
| **溢出（overfit）** | 本阶段只做 **CASE 1（无 overfit，单 step）**。默认配置容量极大（INP/ACC/OUT 各 128 块、WGT 1024 块），32×32 远不溢出。strategy_1..4（多 step）留作后续增量。 |
| **地址分配** | 本阶段**不**引入独立 dram_allocation pass；沿用第一阶段的固定缓冲逻辑基址（INP=64/WGT=8/ACC=192/OUT=256/UOP=5120），由 lowering 按块布局**计算每块偏移**。完整页对齐多层分配 pass 留作后续增量。 |
| **依赖信号量** | 本阶段实现 **4 计数器信号量算法**（LD↔CMP↔ST），覆盖单 step 多块；它本就是上游通用机制，借此一步到位、为多 step 铺路。 |
| **ALU** | 不做（纯 GEMM）。 |
| **验收** | `fsim_only` 为门（结果矩阵 == 上游黄金）；字节级 `cmp` 为强信号。 |

## 3. 架构与数据流

```
func @main(tensor<32x32> %A,%B)        ← 入口（tensor 级 linalg.matmul + fill 初始化 ACC=0）
  │  bufferization 栈                   （→ memref<32x32xi32>，无需 linalg-tile）
  │  --convert-linalg-to-vta            （linalg.matmul(memref<32x32>) → vta.gemm(memref<32x32> operand)）
  │  --lower-vta-gemm                   （通用化：块调度 + uop 表 + 合并 LOAD + 单 GEMM + 逐块 STORE + NOP + 依赖位）
  ▼
vtaisa.* 指令流 ── vta-translate ──► instructions.bin / uop.bin
                 vta-translate --emit-data ──► 多块 input/weight/accumulator/out_init.bin + CSV
                                                      │
                                                      ▼  run_fsim
                                                    FSIM 4 块结果矩阵
```

## 4. `vta.gemm` op 通用化

- operand 仍为 `ins(%lhs, %rhs) outs(%acc)`，但 verifier 放宽：三者均为 2D `memref`，每维都是 **16 的倍数**，且形状满足矩阵乘约束
  （`lhs: M×K`、`rhs: K×N`、`acc: M×N`）。本阶段先支持方阵 32×32×32；矩形（M≠N≠K，仍 16 倍数）作为同一 verifier 的自然延伸。
- `m,n,k`（以块计：`M/16, N/16, K/16`）由 operand 形状推导。
- ODS verifier：拒绝非 16 倍数 / rank≠2 / 维度不匹配。第二阶段「仅 16×16」的硬断言被本阶段放宽（这是计划中的「通用化阶段放开 verifier」）。

## 5. 块调度（lowering 内部，复刻上游 `matrix_partitioning` CASE 1 + `get_gemm_operations`）

以块为单位、行主序索引 `idx = row*blocks_col + col`：
- `A_blocks_row=M/16, A_blocks_col=K/16`；`B_blocks_col=N/16`；`C_blocks_row=M/16, C_blocks_col=N/16`。
- 单 step：`load_A=[0..nbA)`、`load_B=[0..nbB)`、`load_X=load_C=[0..nbC)`。
- GeMM 列表（`get_gemm_operations`）：遍历 `load_A` 的每个 `A[i,k]`，对所有 `B[k,j]` 产出 `(C[i*Ncol+j], A[i*Kcol+k], B[k*Ncol+j])`。
  - 32×32 产出 8 条（见 spike §GeMM 顺序），与黄金 uop.bin 顺序一致。

### 5.1 UOP 表

每条 GeMM op 一条 uop（编译器路径，不合并循环）：
- `dst_idx = c_sram_block_idx × 16`、`src_idx = a_sram_block_idx × 16`、`wgt_idx = b_sram_block_idx`（不乘 16）。
- `*_sram_block_idx` = 该块 ID 在对应 `load_*` 列表中的下标（CASE 1 下即块 ID 本身）。
- 表首再加 1 条 reset 占位 uop `(0,0,0)`（供 reset GEMM 的 `uop[0,1)` 使用）。32×32 → 1+8=9 条。

### 5.2 vtaisa 指令骨架（CASE 1，与第一阶段 16×16 同构）

| 指令 | 通用化取值（块数 = nbA/nbB/nbC，gemm 数 = G） |
|------|----------------------------------------------|
| LOAD UOP（reset） | dram=UOP_base, y=1, x=1 |
| GEMM reset | reset=1, uop[0,1), lp_out=1, lp_in=16, dst_factor_out=16, dst_factor_in=1 |
| LOAD INP | dram=INP_base, **y=nbA**, x=16, stride=16 |
| LOAD WGT | dram=WGT_base, **y=nbB**, x=1, stride=1 |
| LOAD ACC | dram=ACC_base, **y=nbC**, x=16, stride=16 |
| LOAD UOP（gemm） | dram=UOP_base+1, y=1, **x=G**, stride=G |
| GEMM | reset=0, **uop[0,G)**, lp_out=1, lp_in=16, dst_factor_in=1, src_factor_in=1 |
| STORE OUT ×nbC | 逐块：sram=blk×16, dram=OUT_base+blk×16, y=1, x=16, stride=16 |
| NOP-LOAD（INP，y=x=0） | termination |
| NOP-COMPUTE（UOP，y=x=0） | termination |
| FINISH | |

> 合并规则：LOAD 用 2D 形态（`y_size=块数`）一条搞定（块逻辑地址等间距，间距=16 for INP/ACC/OUT、=1 for WGT）。
> STORE **不**合并（上游逐块发），故 nbC 条。32×32 → 14 条指令，与黄金一致。

## 6. 依赖信号量推导（4 计数器算法，复刻上游）

维护四个计数器（初值 0）：`LD->CMP`、`CMP->LD`、`CMP->ST`、`ST->CMP`。
按指令的阶段（LOAD 属 Load 阶段，UOP/ACC LOAD/GEMM 属 Compute 阶段，STORE 属 Store 阶段）在发射时设置 push/pop 位并更新计数器：

- **生产 token**：`push_next`（向下游）/`push_prev`（向上游）置 1 时对应计数器 +1。
- **消费 token**：`pop_*` 置 1 时对应计数器 -1。
- 规则（与上游 `step_load/compute/store` + `termination_sequence` 一致，单 step 推演见 spike §2）：
  - reset GEMM：`push_prev=1`（→ `CMP->LD=1`，告诉 Load「可以开始」）。
  - 首条数据 LOAD（INP）：`pop_next=1`（消费 `CMP->LD`）。
  - 末条数据 LOAD（WGT）：`push_next=1`（→ `LD->CMP=1`）。
  - ACC LOAD：`pop_prev=1`（CASE 1 由上游序定，见黄金）。
  - GEMM：`pop_prev=1`?/`push_prev=1`/`push_next=1`（消费 `LD->CMP`，产 `CMP->LD`、`CMP->ST`）。
  - 首 STORE：`pop_prev=1`（消费 `CMP->ST`）；末 STORE：`push_prev=1`（→ `ST->CMP`）。
  - termination：若 `CMP->LD>0` 发 NOP-LOAD（`pop_next=1`，`push_next=(LD->CMP==0)`）；NOP-COMPUTE（`pop_prev=(LD->CMP>0)`，`pop_next=(ST->CMP>0)`）；FINISH。

> **验证锚点：** 实现产出的 14 条指令 dep 位必须与 spike §解码表逐位一致。先以单 step 的具体序列对齐黄金，再抽象为通用计数器逻辑（二者结果须相同）。

## 7. 多块数据 / CSV 发射器（`VTADataEmitter` 通用化）

输入：原始 N×N 行主序 `input.bin`/`weight.bin`（如 `input_32x32.bin`）。输出（均已用脚本核验对齐黄金）：

| 文件 | 规则 |
|------|------|
| `input.bin` | **块主序、不转置**：块顺序 (r,c) 行主序遍历，每块 16×16 行主序拼接 |
| `weight.bin` | **块主序、每块转置**：同序，但每个 16×16 块写出前 `.T` |
| `accumulator.bin` / `out_init.bin` | `nbC × 256` 个 int32 零 |
| `metadata.csv` | `BS,16,16,True` + `A/X/Y/C` 维度行（见黄金 `metadata.csv`） |
| `memory_addresses.csv` | 固定 6 行：INP/WGT/ACC/OUT/UOP/INSN 的物理/逻辑基址（本阶段沿用固定基址） |
| `layers_name.csv` | 复用（单层） |

## 8. operand → DRAM 缓冲映射（沿用第一/二阶段固定基址）

| operand | 缓冲 | 逻辑基址 | 每块偏移 |
|---------|------|----------|----------|
| `%lhs` (M×K) | INP | 64 | +16/块 |
| `%rhs` (K×N) | WGT | 8 | +1/块 |
| `outs` (M×N) | ACC/OUT | 192 / 256 | +16/块 |
| UOP | UOP | 5120 | +1/uop |

> 这些基址与第一阶段 16×16 相同（页对齐 + 同对象顺序 + 单层 < 1 页），故 32×32 直接复用；通用化阶段后续再引入真正的 allocation pass。

## 9. 数值约定边界

沿用第二阶段：`linalg.matmul` 为被识别标记；矩阵乘语义（`A·B` vs VTA `INP·WGTᵀ` + WGT 每块磁盘转置）由数据发射器的「每块转置」保证与上游一致。本阶段仍以「复用上游数据布局 + 块调度 ⇒ 与上游 FSIM 结果一致」立论。

## 10. 验证策略（验收 = FSIM 结果一致）

| 层级 | 手段 |
|------|------|
| 数据 bin 字节级 | `vta-translate --emit-data` 产出 `input/weight/accumulator.bin` → `cmp` 黄金（**可独立先验**） |
| 指令/UOP 字节级 | lowering+translate → `cmp` `test/golden/matmul_32x32/{instructions,uop}.bin`（强信号） |
| 逐 pass IR | `vta-opt ... | grep -q`（无 FileCheck） |
| 端到端（验收门） | `matmul_32x32.mlir` → 完整管道 → `vta-translate` → FSIM 结果矩阵 == `test/golden/matmul_32x32/fsim_result_32x32.txt` |

## 11. 明确不做（留待后续增量）

- overfit（strategy_1..4、多 step、SRAM 复用调度）；
- 独立 dram_allocation pass（页对齐、多层 base 递增）；
- 非方阵的全面覆盖（先 32×32 方阵，矩形作 verifier 自然延伸但不强验）；
- ALU/卷积 lowering、onnx-mlir 前端。

## 12. 交付物清单

- `include/.../VTA/VTAOps.td`：`vta.gemm` verifier 放宽到「16 倍数 + 维度匹配」。
- `lib/Transforms/ConvertLinalgToVTA.cpp`：放宽到接受 N×N `linalg.matmul`。
- `lib/Transforms/LowerVTAGemm.cpp`：通用化块调度 + uop + 合并 LOAD + 单 GEMM + 逐块 STORE + 4 计数器依赖位（或拆出 `GemmScheduler`/`SemaphoreDeriver` 辅助）。
- `lib/Target/VTADataEmitter.cpp`：多块块主序 + WGT 每块转置 + nbC 块零 ACC/OUT。
- 入口 `test/Target/matmul_tensor_32x32.mlir`；黄金 `test/golden/matmul_32x32/*`（Task 0 已生成）。
- `scripts/run_fsim_linalg_32x32.sh`：端到端 + 结果矩阵自校验。
- 更新 `DESIGN_cn.md`（§2.1 阶段三状态、§7 通用化 lowering、§10）。
