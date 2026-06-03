# Phase-3 Task 0: Generalized GEMM Spike — 32×32×32 golden (from upstream Python)

目标：用上游 Python 参考编译器 `main_vta_compiler.py` 为一个 **32×32×32**（各维 2 块）的纯 GEMM
生成字节级黄金参考，作为 MLIR 侧通用化实现的对照目标。本任务**不改 MLIR 代码**。

## 决策：以 32×32×32 为通用化第一个验收目标

- 32×32 → 以 16×16 块为单位是 **2×2 块**：`nb_A=nb_B=nb_C=4`，`A_blocks_col=B_blocks_col=C_blocks_col=2`。
- 默认配置下片上容量极大（INP/ACC/OUT 各 128 块、WGT 1024 块），故 **CASE 1（无 overfit）→ 单 step、一次 load 全部块**。
- 这是"能强迫出真实多块 tiling + K 维 reduce + 多块地址布局 + 多 STORE"的**最小**用例，且结构与第一阶段 16×16 高度平行，适合作为通用化的第一步。
- 后续 overfit（strategy_1..4，多 step）单独留作 Task 之后的扩展。

## 生成方式（可复现）

VTA IR：`standalone-vta/examples/vta_ir/matmul_32x32.json`（已创建，纯 GEMM，无 ALU/bias）：

```json
{ "NAME":"", "MATRICES":{ "A":[32,32,"../compiler_output/input_32x32.bin"],
  "B":[32,32,"../compiler_output/weight_32x32.bin"], "C":[32,32,"output"] },
  "LOAD":{ "INP":["A"], "WGT":["B"] }, "GEMM":["C","A","B"], "STORE":{ "C":["C"] } }
```

```bash
cd /mnt/c/MLIR-VTA/standalone-vta
python3 -c "import numpy as np; np.random.seed(42)
[np.random.randint(-8,8,size=(32,32),dtype=np.int32).tofile(f'compiler_output/{n}_32x32.bin') for n in ('input','weight')]"
cp examples/vta_ir/matmul_32x32.json compiler_output/
cd examples
python3 ../src/compiler/vta_compiler/main_vta_compiler.py True True False \
    ../config/vta_config.json ../compiler_output/matmul_32x32.json
```

编译器摘要：`1 step, 9 UOPs, 14 instructions`，FSIM `gemm_counter=128`（8 GeMM × 16）。

黄金产物已收入 `mlir-vta/test/golden/matmul_32x32/`：
`instructions.bin`(224B/14), `uop.bin`(36B/9), `input.bin`/`weight.bin`/`accumulator.bin`(各 4096B),
`memory_addresses.csv`, `metadata.csv`, 原始 `input_32x32.bin`/`weight_32x32.bin`, `fsim_result_32x32.txt`(4 块结果矩阵), `matmul_32x32.json`。

## 解码后的 14 条指令（字节级目标）

```
I 0 LOAD  UOP sram=0  dram=5120 y=1 x=1  stride=1                                  | (无 dep)              ← reset uop @ uop_logic_base(0x1400)
I 1 GEMM  reset=1 uop[0,1) lp_out=1 lp_in=16 dst_o=16 dst_i=1                       | push_prev=1           ← ACC 清零；启动 LD↔CMP 流水
I 2 LOAD  INP sram=0  dram=64  y=4 x=16 stride=16                                   | pop_next=1            ← 4 个 INP 块合并为单条 (y=4)
I 3 LOAD  WGT sram=0  dram=8   y=4 x=1  stride=1                                    | push_next=1           ← 4 个 WGT 块
I 4 LOAD  ACC sram=0  dram=192 y=4 x=16 stride=16                                   | pop_prev=1            ← 4 个 ACC 块（零初值）
I 5 LOAD  UOP sram=0  dram=5121 y=1 x=8 stride=8                                    | (无 dep)              ← 8 条 GeMM uop @ base+1
I 6 GEMM  reset=0 uop[0,8) lp_out=1 lp_in=16 dst_i=1 src_i=1                        | push_prev=1 push_next=1 ← 8 uop 串行
I 7 STORE OUT sram=0  dram=256 y=1 x=16 stride=16                                   | pop_prev=1            ← C 块 0
I 8 STORE OUT sram=16 dram=272 y=1 x=16 stride=16                                   | (无 dep)              ← C 块 1
I 9 STORE OUT sram=32 dram=288 y=1 x=16 stride=16                                   | (无 dep)              ← C 块 2
I10 STORE OUT sram=48 dram=304 y=1 x=16 stride=16                                   | push_prev=1           ← C 块 3
I11 LOAD  INP NOP(y=x=0) dram=0                                                     | pop_next=1 push_next=1 ← termination NOP-LOAD
I12 LOAD  UOP NOP(y=x=0) dram=0                                                     | pop_prev=1 pop_next=1  ← termination NOP-COMPUTE
I13 FINISH                                                                         | (无 dep)
```

> 注意：STORE 未被合并（与 LOAD 不同），上游对 4 个 OUT 块逐块发 STORE，sram_base 递增 16、dram 递增 16。
> 依赖位仅在 I7（首条 STORE）pop_prev、I10（末条 STORE）push_prev。

## 9 条 UOP（`uop.bin`，每条 `dst_idx[0,11) src_idx[11,22) wgt_idx[22,32)`）

```
UOP0: dst=0  src=0  wgt=0     ← reset 序列占位（I1 reset gemm 用 uop[0,1)）
UOP1: dst=0  src=0  wgt=0     ← C00 += A00·B00
UOP2: dst=16 src=0  wgt=1     ← C01 += A00·B01
UOP3: dst=0  src=16 wgt=2     ← C00 += A01·B10
UOP4: dst=16 src=16 wgt=3     ← C01 += A01·B11
UOP5: dst=32 src=32 wgt=0     ← C10 += A10·B00
UOP6: dst=48 src=32 wgt=1     ← C11 += A10·B01
UOP7: dst=32 src=48 wgt=2     ← C10 += A11·B10
UOP8: dst=48 src=48 wgt=3     ← C11 += A11·B11
```

- `dst_idx = c_sram_block_idx × 16`、`src_idx = a_sram_block_idx × 16`、`wgt_idx = b_sram_block_idx`（**不乘 16**）。
- I6 GEMM 用 `uop[0,8)`（这里 uop_bgn 从 0 起，但实际 8 条是 uop.bin 索引 1..8；reset 用索引 0）。
  上游把 reset uop 与 gemm uop 都放进同一张表，I6 的 `uop_bgn/uop_end` 是相对 LOAD UOP I5 装入 SRAM 后的局部索引——
  **MLIR 实现以「I5 装 8 条、I6 用 [0,8)」的最终字节为准**（见 §字节锚点）。

### GeMM op 顺序（来自 `get_gemm_operations`，行主序 idx=row×2+col）

按 `load_A=[0,1,2,3]` 遍历，对每个 `A[i,k]` 找所有 `B[k,j]`：
`(C0,A0,B0) (C1,A0,B1) (C0,A1,B2) (C1,A1,B3) (C2,A2,B0) (C3,A2,B1) (C2,A3,B2) (C3,A3,B3)`
其中 A0=(0,0) A1=(0,1) A2=(1,0) A3=(1,1)；B0=(0,0) B1=(0,1) B2=(1,0) B3=(1,1)；C 同理。

## 地址（`memory_addresses.csv`，4KiB 页对齐两遍分配）

```
INP  phys 0x1000  logical 0x40   (= 64)
WGT  phys 0x2000  logical 0x8    (= 8)
ACC  phys 0x3000  logical 0xc0   (= 192)
OUT  phys 0x4000  logical 0x100  (= 256)
UOP  phys 0x5000  logical 0x1400 (= 5120)
INSN phys 0x6000  logical 0x600  (= 1536)
```

- 与第一阶段 16×16 的逻辑基址**完全相同**（64/8/192/256/5120）——因为页对齐 + 同样的对象顺序，且单层数据 < 1 页。
- 每块逻辑偏移：INP/ACC/OUT 每块 +16（16 向量/块），WGT 每块 +1（按块计 logical_divisor=block²）。
  故多块 LOAD 的 `dram_base` = 基址 + 块序×偏移，可用 2D 形态（y_size=块数, x_size/ stride）合并。

## 数据块布局（已用脚本核验，见 spec §数据发射器）

- `input.bin` = **块主序、不转置**：块顺序 (0,0),(0,1),(1,0),(1,1)，每块 16×16 行主序。
- `weight.bin` = **块主序、每块转置**：同样块顺序，但每个 16×16 块写出前 `.T`。
- `accumulator.bin` = 原始 X 矩阵（此处全 0），4096B。

## 给下游实现的关键事实

1. CASE 1 单 step 的指令骨架与第一阶段 16×16 **同构**，差异仅：LOAD `y_size=块数`；UOP LOAD `x_size=gemm数`；GEMM `uop_end=gemm数`；STORE 逐块（块数条）；NOP 终止 3 条。
2. dep 位规律（单 step 纯 GEMM）：reset GEMM push_prev=1；INP(首条 LOAD) pop_next=1；最后一条数据 LOAD（此处 WGT）push_next=1；ACC LOAD pop_prev=1；GEMM push_prev=1 push_next=1；首 STORE pop_prev=1；末 STORE push_prev=1；NOP-LOAD pop_next=1 push_next=1；NOP-COMPUTE pop_prev=1 pop_next=1。
3. 此 case **无需 strategy_1..4、无需多 step**，是验证「tiling→多块 vta.gemm→地址分配→依赖位→多块数据」链路的最小闭环。
