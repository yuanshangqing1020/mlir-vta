# MLIR-VTA

基于 MLIR 的 VTA 加速器编译器，目标是逐步替换 `standalone-vta` 的 Python 两阶段编译器。
本仓库与上游 `standalone-vta/`（VTA ISA、功能/周期仿真器、Python 参考编译器）平级，只负责 **MLIR 侧的编译器**。

> 文档索引见 [`docs/README.md`](docs/README.md)。阶段一规格见 [`docs/specs/phase1-gemm-design.md`](docs/specs/phase1-gemm-design.md)；
> 阶段二 linalg 入口见 [`docs/specs/phase2-linalg-entry-design.md`](docs/specs/phase2-linalg-entry-design.md)。

## 当前状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| 阶段一 | ✅ 完成 | 手写 `vtaisa.*` / `vta.gemm` → 二进制，与 Python 编译器 **字节级一致**，FSIM 验证 |
| 阶段二 | ✅ 完成（16×16 单块） | `linalg.matmul`(tensor) → tile/bufferize → `vta.gemm` → lower → translate → FSIM，结果矩阵与阶段一一致 |
| 阶段三 | 🚧 进行中（通用 GEMM·32×32 多块） | `vta.gemm` 放宽到 16 倍数维度；通用化 `lower-vta-gemm`（块调度 + 多块数据 + 逐块 STORE + 依赖位）；32×32 端到端结果矩阵与上游黄金一致，`instructions/uop/数据 bin` 字节级一致 |

尚未实现（后续增量）：overfit 多 step（strategy_1..4）、依赖信号量通用推导 pass、独立地址分配 pass、ALU/卷积 lowering、ONNX 前端（`onnx-mlir`）、整网。

## Dialect 与操作

编译器定义两个 dialect，分别对应高层算子与 VTA ISA 指令。

### 高层 `vta` dialect

| 操作 | 语法 | 说明 |
|------|------|------|
| `vta.gemm` | `vta.gemm ins(%lhs, %rhs : memref<16x16xi32>, memref<16x16xi32>) outs(%acc : memref<16x16xi32>)` | 单块 16×16 GEMM：`acc += lhs * rhs`（原地累加在 `%acc` 上）。**无显式 `m,n,k` 属性**，尺寸由 operand 形状推导；ODS verifier 拒绝非 16×16 memref。operand 在本阶段是信息性的（供未来地址分配 pass 使用），`lower-vta-gemm` 仍用硬编码 DRAM 地址生成指令。 |

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
| `vtaisa.alu_insn` | 同 `gemm_insn` 因子字段 + `alu_opcode`, `use_imm`, `imm` | ALU 微操作循环（**已定义、尚未有 lowering**） |
| `vtaisa.finish` | （无） | 程序结束标记 |

**16×16 单块 GEMM 的固定序列**（由 `-lower-vta-gemm` 生成）：`uop_table` + **11 条** ISA 指令 + `finish` = 共 13 个 op。示例见 `test/Target/gemm16x16.mlir`。

### 编译 Pass（`vta-opt`）

| Pass | 参数 | 作用 |
|------|------|------|
| `ConvertLinalgToVTA` | `--convert-linalg-to-vta` | 已 bufferize 的 `linalg.matmul`（memref 语义）→ `vta.gemm`；若仍是 tensor 语义则报错 |
| `LowerVTAGemm` | `--lower-vta-gemm` | 单块 `vta.gemm` → 上述 13 个 `vtaisa.*` op |

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
# 展开 vta.gemm 为 13 个 vtaisa op（含 uop_table + finish）
~/mlir-vta-build/bin/vta-opt -lower-vta-gemm test/Target/lower_gemm.mlir -o /tmp/lowered.mlir
# 发射二进制
~/mlir-vta-build/bin/vta-translate /tmp/lowered.mlir -o /tmp/out
ls /tmp/out   # instructions.bin  uop.bin
```

### 2) 阶段一：直接发射手写低层 MLIR + 数据/CSV

```bash
~/mlir-vta-build/bin/vta-translate test/Target/gemm16x16.mlir -o /tmp/out \
  --emit-data --input test/golden/input_16x16.bin --weight test/golden/weight_16x16.bin
```

### 3) 阶段二：`linalg.matmul`(tensor) 完整管道

```bash
~/mlir-vta-build/bin/vta-opt test/Target/matmul_tensor.mlir \
  --linalg-tile="linalg-tile-sizes=16,16,16" \
  --linalg-bufferize --tensor-bufferize --tensor-constant-bufferize \
  --func-bufferize --finalizing-bufferize \
  --convert-linalg-to-vta --canonicalize --lower-vta-gemm \
  -o /tmp/lowered.mlir

~/mlir-vta-build/bin/vta-translate /tmp/lowered.mlir -o /tmp/out \
  --emit-data --input test/golden/input_16x16.bin --weight test/golden/weight_16x16.bin
```

> 入口 IR 用 `constant 0 : i32`（非 `arith.constant`），因本机 `vta-opt` 未注册 `arith` dialect。

### 4) 字节级回归（对比黄金参考）

```bash
cmp /tmp/out/instructions.bin test/golden/instructions.bin && echo "INSN OK"
cmp /tmp/out/uop.bin          test/golden/uop.bin          && echo "UOP OK"
```

### 5) 重新生成黄金参考（固定随机种子）

```bash
scripts/make_golden.sh        # 调用 standalone-vta 的 Python 编译器，刷新 test/golden/
```

### 6) FSIM 端到端验证

```bash
scripts/run_fsim.sh              # 阶段一：gemm16x16.mlir → FSIM
scripts/run_fsim_linalg.sh       # 阶段二：matmul_tensor.mlir 完整管道 → FSIM + 结果矩阵自校验
scripts/run_fsim_linalg_32x32.sh # 阶段三：matmul_tensor_32x32.mlir 多块管道 → FSIM + 4 块结果矩阵自校验
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
│   ├── Dialect/            # round-trip（vta.gemm ins/outs 格式）
│   ├── Target/             # gemm16x16.mlir / lower_gemm.mlir / matmul_tensor.mlir
│   └── golden/             # 字节级黄金参考 fixtures + fsim_result_16x16.txt
├── scripts/                # make_golden.sh / run_fsim.sh / run_fsim_linalg.sh
└── docs/                   # README.md、DESIGN_cn.md、plans/、specs/
```

## 测试约定

本环境未安装 `FileCheck`，故 `.mlir` 顶部 `// RUN: ... FileCheck` 行仅作文档注释；
round-trip 校验用 `vta-opt %s | vta-opt` + `grep`，字节级校验用 `cmp`（见上）。
