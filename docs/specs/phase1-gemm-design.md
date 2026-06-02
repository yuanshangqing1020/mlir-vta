# MLIR-VTA 第一阶段设计 · 16×16 GEMM 最小闭环

> **文档定位：** 本文是第一阶段的**设计规格（spec）**，描述阶段一要交付的架构决策、IR 形态、发射器契约与验收标准；可执行步骤见 [`../plans/phase1-gemm.md`](../plans/phase1-gemm.md)。
> - 架构总览与实现细节展开见 [`../DESIGN_cn.md`](../DESIGN_cn.md)；
> - 阶段二（linalg 入口）见 [`phase2-linalg-entry-design.md`](phase2-linalg-entry-design.md)。
> - 若代码与本文不一致，**以源码为准**，并同步更新本文。

---

## 1. 目标（一句话）

用 C++ MLIR 搭建 **`vtaisa` 低层 dialect + 二进制/数据发射器 + 单块 `vta.gemm` lowering**，在固定 16×16 GEMM 用例上产出与现有 Python 编译器**字节一致**的 `instructions.bin` / `uop.bin` / 数据 bin / CSV，并由现有 FSIM 消费验证。

## 2. 范围决策

| 维度 | 决策 |
|------|------|
| **范围切法** | 只做**单 16×16 块**、**单算子**（GEMM），不碰整网、不碰通用维度 tiling。 |
| **抽象分层** | **两个 dialect**：`vta`（高层算子，入口）与 `vtaisa`（与 128-bit ISA / 32-bit UOP 一一对应，可发射）。 |
| **依赖信号量** | **不推导**：11 条指令的 `push_*`/`pop_*` 位照抄 Python 黄金输出（含 I8/I9 NOP 排空）。 |
| **地址** | **硬编码** `dram_base` 与 `memory_addresses.csv` 布局，不实现 `dram_allocation` pass。 |
| **验收标准** | **字节级 `cmp`**（指令 176 B、UOP 8 B、数据 bin、CRLF 的 CSV）+ FSIM 结果矩阵与 Python 路径一致。 |
| **下游** | FSIM/TSIM **不改**；MLIR-VTA 只替换 Python 编译器的**产物格式**。 |

## 3. 架构与数据流

阶段一打通两条等价入口，汇合到同一组二进制：

```
路径 A（低层手写）          路径 B（高层 + lowering）
test/Target/gemm16x16.mlir   test/Target/lower_gemm.mlir
  module 顶层 vtaisa.*         func 内 vta.gemm ins/outs
         │                            │  vta-opt -lower-vta-gemm
         └────────────┬───────────────┘
                      ▼
            vtaisa.uop_table + 11 条指令 + finish
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
  vta-translate              vta-translate --emit-data
  → instructions.bin/uop.bin → input/weight/accumulator/out_init.bin + 3 CSV
         │                         │
         └────────────┬────────────┘
                      ▼
              standalone-vta FSIM（run_fsim.sh）
```

**设计要点：** 路径 A 直接验证发射器与位域；路径 B 验证「高层算子 → 固定展开 → 同一字节」的闭环。阶段二及以后的中端 pass 都复用本阶段锁死的 **lowering 常量** 与 **发射器语义**。

## 4. 入口 IR

### 4.1 低层入口（`test/Target/gemm16x16.mlir`）

`module` 顶层按序列出 `vtaisa.uop_table`、11 条 `load`/`gemm_insn`/`store`、1 条 `finish`。op 为**纯属性、无 SSA operand**（`() -> ()`），字段名与 Python `structures.py` 同名，便于三方对照。

### 4.2 高层入口（`test/Target/lower_gemm.mlir`）

```mlir
func @main(%A: memref<16x16xi32>, %B: memref<16x16xi32>, %C: memref<16x16xi32>) {
  vta.gemm ins(%A, %B) outs(%C)
  return
}
```

- 表示单块 `C += A · B` 的**语义占位**；阶段一 lowering **不读** memref 地址，仍展开为黄金 11 条指令。
- ODS verifier 强制三 operand 均为 `memref<16x16xi32>`（与阶段二 linalg 入口对齐后的形态一致；阶段一交付时即以此作为高层 op 的 IR 契约）。

## 5. Dialect 设计

### 5.1 低层 `vtaisa`

| op | 角色 |
|----|------|
| `vtaisa.uop_table` | 整张 UOP 表（`dst`/`src`/`wgt` 三个等长 `i64` 数组） |
| `vtaisa.load` / `vtaisa.store` | DRAM ↔ SRAM 搬运（opcode 0/1） |
| `vtaisa.gemm_insn` | 计算宏指令（opcode 2） |
| `vtaisa.alu_insn` | ALU 宏指令（opcode 4）；**本阶段仅定义+发射，无 lowering** |
| `vtaisa.finish` | 结束（opcode 3） |

`BufferId` 枚举（UOP/WGT/INP/ACC/OUT）与 ISA `buffer_id` 数值一致。

### 5.2 高层 `vta`

| op | 角色 |
|----|------|
| `vta.gemm` | 单块 16×16 GEMM；由 `-lower-vta-gemm` 展开为 §6 固定序列 |

阶段一**不**引入 `linalg`、`memref.alloc` 自动生成或 `func` 外的中端优化；`vta-opt` 仅注册本项目 dialect 与 pass（阶段二再 `registerAllDialects`）。

## 6. 黄金指令序列（与数据无关）

来源：`examples/vta_ir/matmul_16x16.json` + 固定 fixture，经 `main_vta_compiler.py` 生成；指令序列**与输入数值无关**，故可作字节基线。

| # | 类型 | 要点 |
|---|------|------|
| I0 | LOAD UOP | `dram_base=5120`, `buffer_id=UOP` |
| I1 | GEMM | `reset=1`, `uop_end=1`, `loop_in=16`, `dst_factor_out=16`, `push_prev=1` |
| I2–I4 | LOAD INP/WGT/ACC | `dram_base=64/8/192`，带 `pop_*`/`push_*` 握手 |
| I5 | LOAD UOP | `dram_base=5121` |
| I6 | GEMM | 第二次计算，`push_prev`+`push_next` |
| I7 | STORE OUT | `dram_base=256`, `buffer_id=OUT` |
| I8–I9 | LOAD NOP | `y_size=x_size=0`，仅动依赖 token |
| I10 | FINISH | opcode=3 |

**UOP 表：** 2 条，均为 `(dst,src,wgt)=(0,0,0)` → `uop.bin` 共 8 字节。

完整 16 字节小端 hex 与 CSV 正文见实施计划 [`../plans/phase1-gemm.md`](../plans/phase1-gemm.md) §D。

## 7. 数据布局与 DRAM 映射

| 缓冲 | 逻辑地址 (hex) | 物理地址 (hex) | 数据文件 | 发射规则 |
|------|----------------|----------------|----------|----------|
| INP | 0x40 | 0x1000 | `input.bin` | 原样拷贝 fixture（1024 B） |
| WGT | 0x8 | 0x2000 | `weight.bin` | **转置**写出（`W[r][c]→out[c][r]`） |
| ACC | 0xc0 | 0x3000 | `accumulator.bin` | 全 0 |
| OUT | 0x100 | 0x4000 | `out_init.bin` | 全 0 |
| UOP | 0x1400 | 0x5000 | （在 `uop.bin`） | — |
| INSN | 0x600 | 0x6000 | （在 `instructions.bin`） | — |

**数值语义（文档边界）：** VTA 硬件执行的是 **INP × WGTᵀ**（权重在磁盘上为转置布局）。阶段一不推导数学，只保证与 Python 路径**同一布局 + 同一指令**下 FSIM 结果一致。

**CSV：** `metadata.csv`、`memory_addresses.csv`、`layers_name.csv` 内容与 Python 一致；行尾必须为 **CRLF**（`\r\n`），否则 `cmp` 失败。

## 8. 位域与发射器（`VTABinaryEmitter`）

- 权威 bit 区间与 `standalone-vta/.../structures.py` 一致（见 [`DESIGN_cn.md`](../DESIGN_cn.md) §4）。
- `setBits` + `packMem`/`packGemm`/`packAlu`/`packUop`；显式小端逐字节写出，不依赖宿主端序。
- 遍历：`module.walk` 按 IR 顺序收集 `vtaisa.*`；未 lower 的 `vta.gemm` → **fail-loud**；`uop_table` 三数组不等长 → 报错。

## 9. Lowering（`LowerVTAGemm`）

- Pass：`-lower-vta-gemm`，`OperationPass<ModuleOp>`。
- 对每个 `vta.gemm`：原位插入 `uop_table` + §6 的 11 条 `vtaisa` op + `finish`，再删除 `vta.gemm`。
- **不**分析 memref 别名或地址；字段常量与 `gemm16x16.mlir` 手写版一一对应。

## 10. 工具

| 工具 | 阶段一职责 |
|------|------------|
| `vta-opt` | 注册 `vta`/`vtaisa` + `-lower-vta-gemm`；round-trip 测试 |
| `vta-translate` | `emitBinary`；可选 `--emit-data --input --weight` |

构建：LLVM/MLIR **13.0.0** 预装于 `/usr/local/llvm`；`clang++-13`；build 目录建议 `~/mlir-vta-build`（避免 `/mnt/c` 空间不足）。

## 11. 验证策略

| 层级 | 手段 | 资产 |
|------|------|------|
| dialect round-trip | `vta-opt %s \| vta-opt` + `grep` | `test/Dialect/*.mlir` |
| 发射器字节级 | `cmp` | `test/Target/gemm16x16.mlir` → `test/golden/{instructions,uop}.bin` |
| lowering 端到端 | lower → translate → `cmp` | `test/Target/lower_gemm.mlir` |
| 数据/CSV | `cmp` | `scripts/make_golden.sh`（固定种子 Python 重生） |
| FSIM | 结果矩阵 | `scripts/run_fsim.sh` |

环境无 `FileCheck` 时，`// RUN:` 行仅作文档；以脚本 `grep`/`cmp` 为准。

## 12. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 位域与 Python 漂移 | 字节 golden + 字段名三处同名；`setBits` assert |
| CSV 行尾不一致 | 发射器强制 CRLF |
| 权重未转置 | `VTADataEmitter` 显式转置 + 与 Python `cmp` |
| 误以为 memref 地址会参与 codegen | 文档与 spec 明确：阶段一仅形状/角色；地址硬编码 |
| LLVM 13 API 差异 | 记录 snake_case 访问器、`mlir/Parser.h` 等适配点（见 DESIGN §8.2） |

## 13. 明确不做（留给后续阶段）

- 通用 `m,n,k`、多块 tiling、`matrix_partitioning`；
- 依赖信号量自动推导、ALU lowering pass；
- `linalg` / bufferization / 地址分配 pass；
- `onnx-mlir` 前端、整网 LeNet-5；
- TSIM 数值断言（`doCompare=false` 即可跑通）。

## 14. 交付物清单

- `include/mlir-vta/Dialect/VTAISA/*`：低层 dialect + `BufferId` 枚举。
- `include/mlir-vta/Dialect/VTA/*`：`vta.gemm`。
- `lib/Target/VTABinaryEmitter.cpp`、`VTADataEmitter.cpp`。
- `lib/Transforms/LowerVTAGemm.cpp`。
- `tools/vta-opt`、`tools/vta-translate`。
- `test/Target/gemm16x16.mlir`、`lower_gemm.mlir`、`test/golden/*`。
- `scripts/make_golden.sh`、`scripts/run_fsim.sh`。
- 本文与 [`../plans/phase1-gemm.md`](../plans/phase1-gemm.md)、[`../DESIGN_cn.md`](../DESIGN_cn.md) 同步维护。

## 15. 与阶段二的关系

阶段二在阶段一锁死的 **11 条指令 + 数据布局 + 发射器** 之上增加 `linalg.matmul` 入口与 bufferization 管道；**不修改** lowering 常量即可保持 FSIM 结果与阶段一黄金一致。阶段二对 `vta.gemm` 的 memref 形态与 `--convert-linalg-to-vta` 详见 [`phase2-linalg-entry-design.md`](phase2-linalg-entry-design.md)。
