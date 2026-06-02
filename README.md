# MLIR-VTA

基于 MLIR 的 VTA 加速器编译器，目标是逐步替换 `standalone-vta` 的 Python 两阶段编译器。
本仓库与上游 `standalone-vta/`（VTA ISA、功能/周期仿真器、Python 参考编译器）平级，只负责 **MLIR 侧的编译器**。

> 设计与分阶段路线见 [`docs/plans/2026-06-02-mlir-vta-phase1-gemm.md`](docs/plans/2026-06-02-mlir-vta-phase1-gemm.md)。

## 当前状态：第一阶段完成

打通了「**MLIR → VTA ISA 二进制**」的最小闭环，对 16×16 GEMM 与 Python 编译器输出 **字节级一致**，并通过功能仿真器（FSIM）端到端验证。

已实现：

- **两个 dialect**
  - 高层 `vta`：张量级算子 `vta.gemm`（memref operand 形态 `ins(%lhs, %rhs) outs(%acc)`，`m,n,k` 由 16×16 operand 形状推导）
  - 低层 `vtaisa`：与 128-bit 宏指令 / 32-bit UOP 一一对应的 ISA op：`vtaisa.load`、`vtaisa.store`、`vtaisa.gemm_insn`、`vtaisa.alu_insn`、`vtaisa.finish`、`vtaisa.uop_table`
- **`-lower-vta-gemm` pass**：把单块 `vta.gemm ins(%lhs, %rhs : memref<16x16xi32>, memref<16x16xi32>) outs(%acc : memref<16x16xi32>)` 展开为完整的 11 条 `vtaisa` ISA 指令 + UOP 表（ODS verifier 仅认 16×16）
- **`vta-translate`**：把低层 `vtaisa` MLIR 发射为 `instructions.bin` / `uop.bin`，以及（`--emit-data`）数据 bin 与 `metadata/memory_addresses/layers_name` CSV，全部与 Python 编译器字节一致

尚未实现（后续阶段）：通用维度 GEMM 的 tiling、依赖信号量自动推导、ALU/卷积的 lowering、ONNX 前端（`onnx-mlir`）、整网。

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

### 1) 高层算子 → ISA 指令 → 二进制（端到端）

```bash
# 展开高层 vta.gemm 为 11 条 ISA 指令
~/mlir-vta-build/bin/vta-opt -lower-vta-gemm test/Target/lower_gemm.mlir -o /tmp/lowered.mlir
# 发射二进制
~/mlir-vta-build/bin/vta-translate /tmp/lowered.mlir -o /tmp/out
ls /tmp/out   # instructions.bin  uop.bin
```

### 2) 直接发射手写低层 MLIR + 数据/CSV

```bash
~/mlir-vta-build/bin/vta-translate test/Target/gemm16x16.mlir -o /tmp/out \
  --emit-data --input test/golden/input_16x16.bin --weight test/golden/weight_16x16.bin
```

### 3) 字节级回归（对比黄金参考）

```bash
cmp /tmp/out/instructions.bin test/golden/instructions.bin && echo "INSN OK"
cmp /tmp/out/uop.bin          test/golden/uop.bin          && echo "UOP OK"
```

### 4) 重新生成黄金参考（固定随机种子）

```bash
scripts/make_golden.sh        # 调用 standalone-vta 的 Python 编译器，刷新 test/golden/
```

### 5) FSIM 端到端验证

```bash
scripts/run_fsim.sh           # 产出工件 → 喂给 standalone-vta 的 FSIM → 打印结果矩阵 + profiler
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
│   └── Transforms/         # LowerVTAGemm pass (vta.gemm → vtaisa.*)
├── tools/
│   ├── vta-opt/            # 注册 vta+vtaisa dialect + pass 的 mlir-opt 变体
│   └── vta-translate/      # MLIR → 二进制/数据/CSV
├── test/
│   ├── Dialect/            # round-trip
│   ├── Target/             # gemm16x16.mlir / lower_gemm.mlir
│   └── golden/             # 字节级黄金参考 fixtures
├── scripts/                # make_golden.sh / run_fsim.sh
└── docs/plans/             # 实施计划
```

## 测试约定

本环境未安装 `FileCheck`，故 `.mlir` 顶部 `// RUN: ... FileCheck` 行仅作文档注释；
round-trip 校验用 `vta-opt %s | vta-opt` + `grep`，字节级校验用 `cmp`（见上）。
