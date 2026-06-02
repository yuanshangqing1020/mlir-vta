# MLIR-VTA 编译器重构 · 第一阶段实施计划（16×16 GEMM 最小闭环）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal（一句话）：** 用 C++ MLIR 搭建一个 `vta` dialect 与二进制发射器，对 16×16 GEMM 这个固定用例，产出与现有 Python 编译器**字节一致**的 `instructions.bin` / `uop.bin` / 数据 bin / CSV，并通过现有 FSIM 仿真。

**Architecture（2-3 句）：** 定义一个**低层** `vta` dialect，其 op 与 VTA 128-bit ISA 位域一一对应；写一个 `vta-translate` 工具把这些 op 打包成与 `structures.py` 完全相同的小端二进制；再写一个最小的 `lower-vta-gemm` pass，把高层 `vta.gemm` 单块算子展开成那 11 条固定指令，从而打通「高层算子 → 二进制 → FSIM」的端到端最小链路。

**Tech Stack：** C++17、MLIR/LLVM 13.0.0（已装于 `/usr/local/llvm`）、CMake 3.22、TableGen、lit + FileCheck（随 LLVM 13 提供）、现有 standalone-vta Python 工具链（仅作黄金参考与回归对照）。

---

## 仓库布局与 Git 命令位置（重要）

- **Git 仓库根 = `/mnt/c/MLIR-VTA/mlir-vta/`**，特性分支 `feat/mlir-vta-phase1`。`standalone-vta/` 是独立的上游仓库，不纳入本仓库。
- 计划中所有形如 `mlir-vta/<x>` 的路径，等价于仓库内的 `<x>`；若命令的工作目录是工作区根 `/mnt/c/MLIR-VTA`，`mlir-vta/<x>` 可直接解析，**无需修改**。
- **唯一需要注意的是 Git 命令**：必须在仓库根内执行。凡计划里写 `cd /mnt/c/MLIR-VTA && git add mlir-vta/... && git commit ...` 的，统一改为：
  ```bash
  cd /mnt/c/MLIR-VTA/mlir-vta && git add -A && git commit -m "<message>"
  ```
- build 目录仍在 `~/mlir-vta-build`（不占 `/mnt/c`，且天然不在仓库内）。

---

## A. 总体路线图（四阶段，仅第一阶段在本计划详述）

| 阶段 | 目标 | 是否本计划范围 |
|------|------|----------------|
| **阶段一（本计划）** | `vta` dialect 低层 op + 二进制/数据/CSV 发射器 + 单块 `vta.gemm` 展开 pass；16×16 GEMM 字节级复刻 + FSIM 通过 | ✅ |
| 阶段二 | `linalg.matmul → vta.gemm` 的 16×16 tiling lowering；`bufferization` 对齐 `dram_allocation`；通用 GEMM | ❌ |
| 阶段三 | 依赖信号量自动配对 pass（Load/Compute/Store token）；ALU（ReLU/池化）；大矩阵 `matrix_partitioning` 等价的 tiling 策略 | ❌ |
| 阶段四 | 接入 `onnx-mlir` 前端：`QLinearConv` 等 → `linalg` → `vta`；整网（LeNet-5）端到端，替换 Python 工具链 | ❌ |

第一阶段刻意**只做低层 op + 发射器 + 单块展开**，把「依赖信号量生成」「通用 tiling」这两块难点推迟到阶段二/三。这样能在最短路径上拿到一个可字节验收的成果，并把发射器正确性与编解码逻辑彻底锁死。

---

## B. 第一阶段范围与验收标准

**做什么：**
1. C++ MLIR 工程骨架（CMake 找到 LLVM 13，构建 `vta-opt`、`vta-translate` 两个工具）。
2. `vta` dialect：低层 op（`vta.load` / `vta.store` / `vta.gemm_insn` / `vta.alu_insn` / `vta.finish`）+ UOP 表示，属性字段与 ISA 位域一一对应。
3. `vta-translate`：把一个 MLIR module 发射为 `instructions.bin` + `uop.bin`（字节级等于黄金参考）。
4. 数据/CSV 发射：`input.bin`、`weight.bin`（转置）、`accumulator.bin`、`out_init.bin`、`metadata.csv`、`memory_addresses.csv`、`layers_name.csv`。
5. 高层 `vta.gemm` 单块算子 + `lower-vta-gemm` pass，展开为黄金参考那 11 条指令 / 2 条 UOP。

**验收（必须全部满足）：**
- `cmp` 比对 `vta-translate` 产出的 `instructions.bin` 与黄金参考 → **0 差异**（176 字节）。
- `cmp` 比对 `uop.bin` → **0 差异**（8 字节）。
- 数据 bin 与 CSV 与 Python 产物 `cmp` 一致（固定输入 fixture 下）。
- 把产物喂给现有 FSIM（`make fsim_compile_single_layer` + `make fsim_single_layer`），打印出的 16×16 结果矩阵与 Python 路径一致。
- `lower-vta-gemm` 从单块 `vta.gemm` 出发，最终二进制也 `cmp` 一致。

**不做（明确排除）：** 通用维度 GEMM、依赖信号量自动推导（第一阶段用黄金参考里的固定依赖位）、ALU 算子的展开 pass、ONNX 前端、TSIM 数值断言（TSIM `doCompare=false`，仅需能跑）。

---

## C. 环境前提（已实测确认）

| 项 | 状态 |
|----|------|
| LLVM/MLIR | **已安装** `/usr/local/llvm`，版本 **13.0.0**；`MLIRConfig.cmake`、`LLVMConfig.cmake`、头文件、`mlir-opt`/`mlir-tblgen`/`FileCheck` 均可用 |
| 编译器 | 系统有 `clang/clang++ 13.0.0` 与 `g++ 9.4.0`。**统一用 `clang++-13`** 以匹配 LLVM 13 的 ABI（避免 libstdc++ ABI 不一致） |
| 构建工具 | `cmake 3.22.1`、`make`（**无 `ninja`** → 用 `Unix Makefiles` generator） |
| 内存/磁盘 | 内存 7.8 GiB；`/mnt/c` 仅剩 ~5 GiB（紧张）、`/`（即 `~`，882 GiB 空闲）→ **build 目录放 `~/mlir-vta-build`，不要放在 `/mnt/c`** |
| Python 参考链路 | `python3 3.8.10` + `numpy 1.22.4` 可直接跑现有编译器生成黄金参考 |

> 关键好消息：**无需从源码编译 LLVM**，省掉数小时构建与数十 GiB 磁盘占用。

**环境校验命令（任何 task 前先跑一次）：**
```bash
/usr/local/llvm/bin/llvm-config --version          # 期望: 13.0.0
ls /usr/local/llvm/lib/cmake/mlir/MLIRConfig.cmake  # 期望: 文件存在
echo 'module {}' | /usr/local/llvm/bin/mlir-opt     # 期望: 打印 "module {\n}"
```

---

## D. 黄金参考（16×16 GEMM，实测自现有 Python 编译器）

> 来源：`examples/vta_ir/matmul_16x16.json` + 随机 int32 输入，跑 `main_vta_compiler.py` 得到。指令序列**与数据值无关**（确定性），故可直接作为字节基线。`config/vta_config.json`：`LOG_BLOCK=4`（块=16）、`LOG_*_WIDTH=5`（int32）。

### D.1 指令序列（11 条，每条 16 字节，小端 hex 为字节倒序后的十六进制）

| # | 类型 | 关键字段 | 16 字节小端 hex |
|---|------|----------|------------------|
| I0 | LOAD UOP | `dram_base=5120, y_size=1, x_size=1, x_stride=1, buffer_id=0(UOP)` | `00000001000100010000005000000000` |
| I1 | GEMM | `push_prev_dep=1, reset=1, uop_end=1, loop_out=1, loop_in=16, dst_factor_out=16, dst_factor_in=1` | `000000000000081000200008002000A2` |
| I2 | LOAD INP | `pop_next_dep=1, buffer_id=2, dram_base=64, y_size=1, x_size=16, x_stride=16` | `00000010001000010000000100000110` |
| I3 | LOAD WGT | `push_next_dep=1, buffer_id=1, dram_base=8, y_size=1, x_size=1, x_stride=1` | `000000010001000100000000200000C0` |
| I4 | LOAD ACC | `pop_prev_dep=1, buffer_id=3, dram_base=192, y_size=1, x_size=16, x_stride=16` | `00000010001000010000000300000188` |
| I5 | LOAD UOP | `dram_base=5121, y_size=1, x_size=1, x_stride=1, buffer_id=0` | `00000001000100010000005004000000` |
| I6 | GEMM | `push_prev_dep=1, push_next_dep=1, uop_end=1, loop_out=1, loop_in=16, dst_factor_in=1, src_factor_in=1` | `00000002000008000020000800200062` |
| I7 | STORE OUT | `pop_prev_dep=1, push_prev_dep=1, buffer_id=4, dram_base=256, y_size=1, x_size=16, x_stride=16` | `00000010001000010000000400000229` |
| I8 | LOAD INP (NOP) | `pop_next_dep=1, push_next_dep=1, buffer_id=2`（y/x_size=0 → 空搬运） | `00000000000000000000000000000150` |
| I9 | LOAD UOP (NOP) | `pop_prev_dep=1, pop_next_dep=1, buffer_id=0`（y/x_size=0） | `00000000000000000000000000000018` |
| I10 | FINISH | `opcode=3`，其余 0 | `00000000000000000000000000000003` |

> 说明：I8/I9 是**收尾排空**用的 NOP（`x_size=y_size=0` 不搬数据，只动依赖 token），I10 是必需的 FINISH。第一阶段直接照抄这些字段（含依赖位），不做自动推导。

### D.2 UOP 表（2 条，每条 4 字节）

| # | dst_idx | src_idx | wgt_idx | 4 字节小端 hex |
|---|---------|---------|---------|----------------|
| UOP0 | 0 | 0 | 0 | `00000000` |
| UOP1 | 0 | 0 | 0 | `00000000` |

### D.3 数据 bin 布局规则（单块 16×16，已实测）

| 文件 | 大小 | 规则 |
|------|------|------|
| `input.bin` | 1024 B | **= 原始 `input_16x16.bin` 行主序，完全相同**（单块无需重排） |
| `weight.bin` | 1024 B | **= 原始 `weight_16x16.bin` 的转置**（`W.T`，行主序写出） |
| `accumulator.bin` | 1024 B | 256 个 int32 **全 0** |
| `out_init.bin` | 1024 B | 256 个 int32 **全 0** |
| `expected_out_sram.bin` | 1024 B | TSIM 对比用；第一阶段 TSIM 不强制断言，可由 Python 生成或暂留空逻辑 |

### D.4 CSV 内容（字节级目标）

`metadata.csv`：
```csv
Matrix (or Block Size),Nb rows,Nb columns,Is it square?
BS,16,16,True
A,16,16,True
X,16,16,False
Y,0,0,True
C,16,16,True
```

`memory_addresses.csv`：
```csv
Buffer type,Physical address (hex),Logical address (hex)
INP,0x1000,0x40
WGT,0x2000,0x8
ACC,0x3000,0xc0
OUT,0x4000,0x100
UOP,0x5000,0x1400
INSN,0x6000,0x600
```

`layers_name.csv`：
```csv
Line identifier,Nb of VTA IR,Provide execution log
nb_vta_ir,1,True
Line identifier,VTA IR name,Last physical DRAM address allocated by the layer
0,,0x60af
```

---

## E. 工程文件结构

新工程放在仓库根的 `mlir-vta/` 子目录（与现有 `standalone-vta/` 平级）。

```
/mnt/c/MLIR-VTA/mlir-vta/
├── CMakeLists.txt                      # 顶层；find_package(MLIR)；C++17；clang++-13
├── README.md                           # 构建/运行说明
├── include/mlir-vta/Dialect/VTA/
│   ├── VTADialect.h
│   ├── VTADialect.td                   # dialect 声明
│   ├── VTAOps.h
│   ├── VTAOps.td                       # 所有 op 定义
│   └── VTAEnums.td                     # buffer_id / alu_opcode 枚举属性
├── include/mlir-vta/Target/
│   └── VTABinaryEmitter.h              # 发射器接口
├── lib/Dialect/VTA/
│   ├── VTADialect.cpp
│   └── VTAOps.cpp                      # op 实现 + verifier
├── lib/Target/
│   ├── VTABinaryEmitter.cpp            # 128-bit/32-bit 位域打包（对齐 structures.py）
│   └── VTADataEmitter.cpp              # 数据 bin + CSV 发射
├── lib/Transforms/
│   └── LowerVTAGemm.cpp                # 单块 vta.gemm → 11 条低层 insn
├── tools/
│   ├── vta-opt/vta-opt.cpp             # 注册 dialect + pass 的 mlir-opt 变体
│   └── vta-translate/vta-translate.cpp # 读 .mlir → 输出目录写 bin/csv
├── test/
│   ├── lit.cfg.py                      # lit 配置
│   ├── Dialect/roundtrip.mlir          # FileCheck round-trip
│   ├── Target/gemm16x16.mlir           # 低层 11 条 op 的手写 MLIR
│   ├── Target/lower_gemm.mlir          # 高层 vta.gemm 单块
│   └── golden/                         # 黄金参考 fixtures
│       ├── instructions.bin            # 176 B（从 D.1 生成）
│       ├── uop.bin                     # 8 B
│       ├── input_16x16.bin             # 固定输入 fixture（去随机）
│       └── weight_16x16.bin
└── scripts/
    ├── make_golden.sh                  # 用 Python 编译器重生成 golden（固定种子）
    └── run_fsim.sh                     # 把 mlir-vta 产物拷进 standalone-vta 跑 FSIM
```

---

## F. 位域打包规范（发射器必须严格遵守）

小端、`_pack_=1`，字段按声明顺序从 **bit 0** 起递增填充。下面给出每种结构体的精确 bit 区间（`[起, 止)`，单位 bit），与 `standalone-vta/src/compiler/vta_compiler/operations_definition/structures.py` 完全一致。

**VTAUop（32 bit，1 个 uint32）：**
```
dst_idx [0,11)  src_idx [11,22)  wgt_idx [22,32)
```

**VTAMemInsn（128 bit，两个 uint64：lo=word0, hi=word1）：**
```
word0: opcode[0,3) pop_prev[3] pop_next[4] push_prev[5] push_next[6]
       buffer_id[7,10) sram_base[10,26) dram_base[26,58) unused[58,64)
word1: y_size[0,16) x_size[16,32) x_stride[32,48)
       y_pad_top[48,52) y_pad_bottom[52,56) x_pad_left[56,60) x_pad_right[60,64)
```

**VTAGemInsn（128 bit）：**
```
word0: opcode[0,3) pop_prev[3] pop_next[4] push_prev[5] push_next[6] reset[7]
       uop_bgn[8,21) uop_end[21,35) loop_out[35,49) loop_in[49,63) unused[63]
word1: dst_factor_out[0,11) dst_factor_in[11,22) src_factor_out[22,33)
       src_factor_in[33,44) wgt_factor_out[44,54) wgt_factor_in[54,64)
```

**VTAAluInsn（128 bit）：**
```
word0: 同 GeMM 前缀（opcode..unused）
word1: dst_factor_out[0,11) dst_factor_in[11,22) src_factor_out[22,33)
       src_factor_in[33,44) alu_opcode[44,47) use_imm[47] imm[48,64)
```

**发射顺序：** `instructions.bin` 按 I0..I10 顺序，每条先写 word0 再写 word1（各 8 字节小端）→ 即直接 `write(&word0,8); write(&word1,8)`，x86 小端下等价于结构体内存布局。`uop.bin` 按 UOP0..UOP1 顺序每条 4 字节小端。

**打包参考实现（C++，发射器核心，可直接用）：**
```cpp
// lib/Target/VTABinaryEmitter.cpp 中的位域打包工具
static inline void setBits(uint64_t &word, unsigned lo, unsigned width, uint64_t val) {
  uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1);
  word |= (val & mask) << lo;
}

// 例：打包一条 GEMM
std::array<uint64_t,2> packGemm(const GemmFields &f) {
  uint64_t w0 = 0, w1 = 0;
  setBits(w0, 0, 3, 2 /*opcode GEMM*/);
  setBits(w0, 3, 1, f.popPrev); setBits(w0, 4, 1, f.popNext);
  setBits(w0, 5, 1, f.pushPrev); setBits(w0, 6, 1, f.pushNext);
  setBits(w0, 7, 1, f.reset);
  setBits(w0, 8, 13, f.uopBgn); setBits(w0, 21, 14, f.uopEnd);
  setBits(w0, 35, 14, f.loopOut); setBits(w0, 49, 14, f.loopIn);
  setBits(w1, 0, 11, f.dstFactorOut); setBits(w1, 11, 11, f.dstFactorIn);
  setBits(w1, 22, 11, f.srcFactorOut); setBits(w1, 33, 11, f.srcFactorIn);
  setBits(w1, 44, 10, f.wgtFactorOut); setBits(w1, 54, 10, f.wgtFactorIn);
  return {w0, w1};
}
```

**单元验证（用 D.1 的 I1 验证 packGemm）：** I1 字段 `push_prev=1, reset=1, uop_end=1, loop_out=1, loop_in=16, dst_factor_out=16, dst_factor_in=1`。期望小端 hex（字节倒序）`000000000000081000200008002000A2`，即正序字节 `A2 00 20 00 08 00 20 00 10 08 00 00 00 00 00 00`，对应 `w0=0x0000000800200​0A2`、`w1=0x0000000000000810`。Task 5 的测试会断言这一点。

---

## G. 任务分解（bite-sized，TDD，频繁提交）

> 约定：所有 `cmake`/构建命令的 build 目录在 `~/mlir-vta-build`（不占 `/mnt/c`）。源码在 `/mnt/c/MLIR-VTA/mlir-vta`。每个 Task 末尾提交一次。

### Task 0：固化黄金参考 fixtures

**Files:**
- Create: `mlir-vta/test/golden/instructions.bin`、`uop.bin`、`input_16x16.bin`、`weight_16x16.bin`
- Create: `mlir-vta/scripts/make_golden.sh`

- [ ] **Step 1：写 `make_golden.sh`（固定种子，确保可复现）**

```bash
#!/usr/bin/env bash
set -euo pipefail
SVTA=/mnt/c/MLIR-VTA/standalone-vta
OUT="$(cd "$(dirname "$0")/../test/golden" && pwd)"
cd "$SVTA"
python3 - <<'PY'
import numpy as np
np.random.seed(42)
for name in ("input","weight"):
    a = np.random.randint(-8, 8, size=(16,16), dtype=np.int32)
    a.tofile(f"compiler_output/{name}_16x16.bin")
print("fixed input/weight written")
PY
cp examples/vta_ir/matmul_16x16.json compiler_output/
cd examples
python3 ../src/compiler/vta_compiler/main_vta_compiler.py False True False \
    ../config/vta_config.json ../compiler_output/matmul_16x16.json >/dev/null
cp ../compiler_output/instructions.bin ../compiler_output/uop.bin \
   ../compiler_output/input_16x16.bin ../compiler_output/weight_16x16.bin "$OUT/"
echo "golden refreshed into $OUT"
```

- [ ] **Step 2：运行并确认产物大小**

Run:
```bash
chmod +x mlir-vta/scripts/make_golden.sh && mlir-vta/scripts/make_golden.sh
ls -l mlir-vta/test/golden/instructions.bin mlir-vta/test/golden/uop.bin
```
Expected: `instructions.bin` 176 字节，`uop.bin` 8 字节。

- [ ] **Step 3：提交**
```bash
cd /mnt/c/MLIR-VTA && git add mlir-vta/scripts mlir-vta/test/golden && \
git commit -m "chore(mlir-vta): pin 16x16 GEMM golden fixtures"
```

---

### Task 1：CMake 工程骨架 + 空 dialect 注册

**Files:**
- Create: `mlir-vta/CMakeLists.txt`
- Create: `mlir-vta/include/mlir-vta/Dialect/VTA/VTADialect.td`、`VTADialect.h`
- Create: `mlir-vta/lib/Dialect/VTA/VTADialect.cpp`
- Create: `mlir-vta/tools/vta-opt/vta-opt.cpp`
- Create: `mlir-vta/test/lit.cfg.py`、`mlir-vta/test/Dialect/empty.mlir`

- [ ] **Step 1：写顶层 `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.20)
project(mlir-vta LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include/mlir-vta/Dialect/VTA)
add_subdirectory(lib/Dialect/VTA)
add_subdirectory(tools/vta-opt)
```

- [ ] **Step 2：写 `VTADialect.td`（dialect 声明）**

```tablegen
#ifndef VTA_DIALECT_TD
#define VTA_DIALECT_TD
include "mlir/IR/OpBase.td"
def VTA_Dialect : Dialect {
  let name = "vta";
  let cppNamespace = "::mlir::vta";
  let summary = "VTA accelerator ISA dialect";
}
class VTA_Op<string mnemonic, list<Trait> traits = []>
    : Op<VTA_Dialect, mnemonic, traits>;
#endif
```

- [ ] **Step 3：写 `include/.../VTA/CMakeLists.txt`（TableGen 规则）**

```cmake
set(LLVM_TARGET_DEFINITIONS VTADialect.td)
mlir_tablegen(VTADialect.h.inc -gen-dialect-decls -dialect=vta)
mlir_tablegen(VTADialect.cpp.inc -gen-dialect-defs -dialect=vta)
add_public_tablegen_target(VTADialectIncGen)
```

- [ ] **Step 4：写 `VTADialect.h` / `VTADialect.cpp`（最小实现）**

`VTADialect.h`:
```cpp
#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir-vta/Dialect/VTA/VTADialect.h.inc"
```
`VTADialect.cpp`:
```cpp
#include "mlir-vta/Dialect/VTA/VTADialect.h"
using namespace mlir; using namespace mlir::vta;
#include "mlir-vta/Dialect/VTA/VTADialect.cpp.inc"
void VTADialect::initialize() {}
```

`lib/Dialect/VTA/CMakeLists.txt`:
```cmake
add_mlir_dialect_library(MLIRVTA
  VTADialect.cpp
  DEPENDS VTADialectIncGen
  LINK_LIBS PUBLIC MLIRIR)
```

- [ ] **Step 5：写 `vta-opt.cpp` 与其 CMake**

`tools/vta-opt/vta-opt.cpp`:
```cpp
#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::vta::VTADialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "vta-opt\n", registry));
}
```
`tools/vta-opt/CMakeLists.txt`:
```cmake
add_llvm_executable(vta-opt vta-opt.cpp)
llvm_update_compile_flags(vta-opt)
target_link_libraries(vta-opt PRIVATE MLIRVTA MLIROptLib MLIRIR MLIRParser MLIRSupport)
mlir_check_all_link_libraries(vta-opt)
```

- [ ] **Step 6：配置 + 构建**

Run:
```bash
cmake -S /mnt/c/MLIR-VTA/mlir-vta -B ~/mlir-vta-build -G "Unix Makefiles" \
  -DMLIR_DIR=/usr/local/llvm/lib/cmake/mlir \
  -DLLVM_DIR=/usr/local/llvm/lib/cmake/llvm \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
  -DCMAKE_BUILD_TYPE=Release
cmake --build ~/mlir-vta-build -j"$(nproc)" --target vta-opt
```
Expected: 生成 `~/mlir-vta-build/bin/vta-opt`。

- [ ] **Step 7：round-trip 空 module**

`test/Dialect/empty.mlir`:
```mlir
// RUN: vta-opt %s | FileCheck %s
// CHECK: module
module {}
```
Run: `~/mlir-vta-build/bin/vta-opt mlir-vta/test/Dialect/empty.mlir`
Expected: 打印 `module {`，退出码 0。

- [ ] **Step 8：提交**
```bash
git add mlir-vta && git commit -m "feat(mlir-vta): scaffold project + empty vta dialect"
```

---

### Task 2：定义低层 ISA op（load/store/gemm_insn/alu_insn/finish）

**Files:**
- Create: `mlir-vta/include/mlir-vta/Dialect/VTA/VTAEnums.td`、`VTAOps.td`、`VTAOps.h`
- Create: `mlir-vta/lib/Dialect/VTA/VTAOps.cpp`
- Modify: `VTADialect.cpp`（注册 op）、各 `CMakeLists.txt`（加 op TableGen）
- Test: `mlir-vta/test/Dialect/roundtrip.mlir`

> 设计：低层 op 用**纯属性**承载所有 ISA 字段（无 SSA 操作数），便于 1:1 发射。所有 op 放在一个 `vta.program` 区域里按顺序排列；UOP 表用一个 `vta.uop_table` 属性数组承载。

- [ ] **Step 1：写 `VTAEnums.td`**

```tablegen
#ifndef VTA_ENUMS_TD
#define VTA_ENUMS_TD
include "mlir/IR/EnumAttr.td"
def VTA_BufUOP : I64EnumAttrCase<"UOP", 0>;
def VTA_BufWGT : I64EnumAttrCase<"WGT", 1>;
def VTA_BufINP : I64EnumAttrCase<"INP", 2>;
def VTA_BufACC : I64EnumAttrCase<"ACC", 3>;
def VTA_BufOUT : I64EnumAttrCase<"OUT", 4>;
def VTA_BufferId : I64EnumAttr<"BufferId", "VTA on-chip buffer",
    [VTA_BufUOP, VTA_BufWGT, VTA_BufINP, VTA_BufACC, VTA_BufOUT]> {
  let cppNamespace = "::mlir::vta";
}
#endif
```

- [ ] **Step 2：写 `VTAOps.td`（5 个低层 op + uop_table）**

```tablegen
#ifndef VTA_OPS_TD
#define VTA_OPS_TD
include "mlir-vta/Dialect/VTA/VTADialect.td"
include "mlir-vta/Dialect/VTA/VTAEnums.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// 公共依赖位
class DepBits {
  dag args = (ins DefaultValuedAttr<I1Attr,"false">:$pop_prev,
                  DefaultValuedAttr<I1Attr,"false">:$pop_next,
                  DefaultValuedAttr<I1Attr,"false">:$push_prev,
                  DefaultValuedAttr<I1Attr,"false">:$push_next);
}

def VTA_LoadOp : VTA_Op<"load"> {
  let arguments = (ins VTA_BufferId:$buffer_id,
    DefaultValuedAttr<I1Attr,"false">:$pop_prev,
    DefaultValuedAttr<I1Attr,"false">:$pop_next,
    DefaultValuedAttr<I1Attr,"false">:$push_prev,
    DefaultValuedAttr<I1Attr,"false">:$push_next,
    I64Attr:$sram_base, I64Attr:$dram_base,
    I64Attr:$y_size, I64Attr:$x_size, I64Attr:$x_stride,
    DefaultValuedAttr<I64Attr,"0">:$y_pad_top,
    DefaultValuedAttr<I64Attr,"0">:$y_pad_bottom,
    DefaultValuedAttr<I64Attr,"0">:$x_pad_left,
    DefaultValuedAttr<I64Attr,"0">:$x_pad_right);
  let assemblyFormat = "attr-dict";
}

def VTA_StoreOp : VTA_Op<"store"> {
  let arguments = (ins VTA_BufferId:$buffer_id,
    DefaultValuedAttr<I1Attr,"false">:$pop_prev,
    DefaultValuedAttr<I1Attr,"false">:$pop_next,
    DefaultValuedAttr<I1Attr,"false">:$push_prev,
    DefaultValuedAttr<I1Attr,"false">:$push_next,
    I64Attr:$sram_base, I64Attr:$dram_base,
    I64Attr:$y_size, I64Attr:$x_size, I64Attr:$x_stride);
  let assemblyFormat = "attr-dict";
}

def VTA_GemmInsnOp : VTA_Op<"gemm_insn"> {
  let arguments = (ins
    DefaultValuedAttr<I1Attr,"false">:$pop_prev,
    DefaultValuedAttr<I1Attr,"false">:$pop_next,
    DefaultValuedAttr<I1Attr,"false">:$push_prev,
    DefaultValuedAttr<I1Attr,"false">:$push_next,
    DefaultValuedAttr<I1Attr,"false">:$reset,
    I64Attr:$uop_bgn, I64Attr:$uop_end, I64Attr:$loop_out, I64Attr:$loop_in,
    DefaultValuedAttr<I64Attr,"0">:$dst_factor_out,
    DefaultValuedAttr<I64Attr,"0">:$dst_factor_in,
    DefaultValuedAttr<I64Attr,"0">:$src_factor_out,
    DefaultValuedAttr<I64Attr,"0">:$src_factor_in,
    DefaultValuedAttr<I64Attr,"0">:$wgt_factor_out,
    DefaultValuedAttr<I64Attr,"0">:$wgt_factor_in);
  let assemblyFormat = "attr-dict";
}

def VTA_AluInsnOp : VTA_Op<"alu_insn"> {
  let arguments = (ins
    DefaultValuedAttr<I1Attr,"false">:$pop_prev,
    DefaultValuedAttr<I1Attr,"false">:$pop_next,
    DefaultValuedAttr<I1Attr,"false">:$push_prev,
    DefaultValuedAttr<I1Attr,"false">:$push_next,
    DefaultValuedAttr<I1Attr,"false">:$reset,
    I64Attr:$uop_bgn, I64Attr:$uop_end, I64Attr:$loop_out, I64Attr:$loop_in,
    DefaultValuedAttr<I64Attr,"0">:$dst_factor_out,
    DefaultValuedAttr<I64Attr,"0">:$dst_factor_in,
    DefaultValuedAttr<I64Attr,"0">:$src_factor_out,
    DefaultValuedAttr<I64Attr,"0">:$src_factor_in,
    I64Attr:$alu_opcode,
    DefaultValuedAttr<I1Attr,"false">:$use_imm,
    DefaultValuedAttr<I64Attr,"0">:$imm);
  let assemblyFormat = "attr-dict";
}

def VTA_FinishOp : VTA_Op<"finish"> { let assemblyFormat = "attr-dict"; }

// UOP 表：一个 op 承载所有 uop（每条 = [dst,src,wgt]）
def VTA_UopTableOp : VTA_Op<"uop_table"> {
  let arguments = (ins I64ArrayAttr:$dst, I64ArrayAttr:$src, I64ArrayAttr:$wgt);
  let assemblyFormat = "attr-dict";
}
#endif
```

- [ ] **Step 3：`VTAOps.h` 引入生成头 + 注册**

`VTAOps.h`:
```cpp
#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir-vta/Dialect/VTA/VTAEnums.h.inc"
#define GET_OP_CLASSES
#include "mlir-vta/Dialect/VTA/VTAOps.h.inc"
```
在 `VTADialect.cpp` 的 `initialize()` 里：
```cpp
void VTADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-vta/Dialect/VTA/VTAOps.cpp.inc"
  >();
}
```
`VTAOps.cpp`:
```cpp
#include "mlir-vta/Dialect/VTA/VTAOps.h"
#include "mlir-vta/Dialect/VTA/VTADialect.h"
using namespace mlir; using namespace mlir::vta;
#include "mlir-vta/Dialect/VTA/VTAEnums.cpp.inc"
#define GET_OP_CLASSES
#include "mlir-vta/Dialect/VTA/VTAOps.cpp.inc"
```

- [ ] **Step 4：更新 TableGen / 库 CMake**

`include/.../VTA/CMakeLists.txt` 追加：
```cmake
set(LLVM_TARGET_DEFINITIONS VTAOps.td)
mlir_tablegen(VTAOps.h.inc -gen-op-decls)
mlir_tablegen(VTAOps.cpp.inc -gen-op-defs)
set(LLVM_TARGET_DEFINITIONS VTAEnums.td)
mlir_tablegen(VTAEnums.h.inc -gen-enum-decls)
mlir_tablegen(VTAEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(VTAOpsIncGen)
```
`lib/Dialect/VTA/CMakeLists.txt`：`VTAOps.cpp` 加入源文件，`DEPENDS` 加 `VTAOpsIncGen`。

- [ ] **Step 5：写 round-trip 测试**

`test/Dialect/roundtrip.mlir`:
```mlir
// RUN: vta-opt %s | vta-opt | FileCheck %s
module {
  // CHECK: vta.uop_table
  "vta.uop_table"() {dst = [0, 0], src = [0, 0], wgt = [0, 0]} : () -> ()
  // CHECK: vta.load
  "vta.load"() {buffer_id = 0 : i64, dram_base = 5120 : i64, y_size = 1 : i64,
                x_size = 1 : i64, x_stride = 1 : i64, sram_base = 0 : i64} : () -> ()
  // CHECK: vta.gemm_insn
  "vta.gemm_insn"() {reset = true, push_prev = true, uop_bgn = 0 : i64,
                     uop_end = 1 : i64, loop_out = 1 : i64, loop_in = 16 : i64,
                     dst_factor_out = 16 : i64, dst_factor_in = 1 : i64} : () -> ()
  // CHECK: vta.finish
  "vta.finish"() : () -> ()
}
```

- [ ] **Step 6：构建并跑测试**

Run:
```bash
cmake --build ~/mlir-vta-build -j"$(nproc)" --target vta-opt
~/mlir-vta-build/bin/vta-opt mlir-vta/test/Dialect/roundtrip.mlir | \
  /usr/local/llvm/bin/FileCheck mlir-vta/test/Dialect/roundtrip.mlir
```
Expected: 退出码 0（FileCheck 全部命中）。

- [ ] **Step 7：提交**
```bash
git add mlir-vta && git commit -m "feat(mlir-vta): add low-level VTA ISA ops + roundtrip test"
```

---

### Task 3：二进制发射器 `vta-translate`（核心，字节级）

**Files:**
- Create: `mlir-vta/include/mlir-vta/Target/VTABinaryEmitter.h`
- Create: `mlir-vta/lib/Target/VTABinaryEmitter.cpp`
- Create: `mlir-vta/tools/vta-translate/vta-translate.cpp` + CMake
- Test: `mlir-vta/test/Target/gemm16x16.mlir`（手写 11 条 op）

- [ ] **Step 1：写位域打包工具（见 F 节 `setBits`/`packGemm`），并补 `packMem`/`packAlu`/`packUop`**

按 F 节 bit 区间逐一实现 4 个 pack 函数，返回 `std::array<uint64_t,2>`（insn）或 `uint32_t`（uop）。

- [ ] **Step 2：写 `emitBinary(ModuleOp, outDir)`**

遍历 module 内 op，按出现顺序：遇 `vta.uop_table` 收集 uop；遇 `vta.load/store/gemm_insn/alu_insn/finish` 调用对应 pack，把 16 字节追加进 `insnBytes`；最后写 `outDir/instructions.bin`、`outDir/uop.bin`。属性读取用 `op->getAttrOfType<IntegerAttr>("...").getInt()`、布尔用 `BoolAttr`。

- [ ] **Step 3：写 `vta-translate.cpp`（CLI：`vta-translate input.mlir -o outdir`）**

用 `parseSourceFile<ModuleOp>` 解析，注册 `vta` dialect，调用 `emitBinary`。

- [ ] **Step 4：手写 `gemm16x16.mlir` = D.1 的 11 条 + D.2 的 2 条 UOP**

按 D.1 表逐条写出 op（属性值照抄，依赖位照抄）。这是发射器的金标准输入。

- [ ] **Step 5：构建 + 字节级断言（关键验收）**

Run:
```bash
cmake --build ~/mlir-vta-build -j"$(nproc)" --target vta-translate
mkdir -p /tmp/vtaout
~/mlir-vta-build/bin/vta-translate mlir-vta/test/Target/gemm16x16.mlir -o /tmp/vtaout
cmp /tmp/vtaout/instructions.bin mlir-vta/test/golden/instructions.bin && echo "INSN OK"
cmp /tmp/vtaout/uop.bin mlir-vta/test/golden/uop.bin && echo "UOP OK"
```
Expected: 打印 `INSN OK` 和 `UOP OK`，`cmp` 无差异输出。**这是第一阶段的核心验收点。**

- [ ] **Step 6：补一个 round-trip 反解校验（可选但推荐）**

用 Python 的 `decode_vta_insn` 解 `/tmp/vtaout/instructions.bin`，与 D.1 字段逐项核对（脚本化）。

- [ ] **Step 7：提交**
```bash
git add mlir-vta && git commit -m "feat(mlir-vta): byte-exact instruction/uop binary emitter"
```

---

### Task 4：数据 bin + CSV 发射器

**Files:**
- Create: `mlir-vta/lib/Target/VTADataEmitter.cpp`（被 `vta-translate` 调用）
- Modify: `vta-translate.cpp`（加 `--emit-data --input <inp16x16.bin> --weight <wgt16x16.bin>` 选项）

- [ ] **Step 1：实现数据搬运（按 D.3 规则）**

`input.bin` = 原样拷贝输入；`weight.bin` = 读 16×16 int32 转置后写出；`accumulator.bin`/`out_init.bin` = 256 个 int32 零。用 `std::ifstream`/`ofstream` 二进制读写。

- [ ] **Step 2：实现 CSV 发射（按 D.4，字符串完全照抄）**

`metadata.csv`、`memory_addresses.csv`、`layers_name.csv` 三个文件内容硬编码为 D.4 中的精确文本（含表头与逗号）。

- [ ] **Step 3：字节级断言**

Run:
```bash
~/mlir-vta-build/bin/vta-translate mlir-vta/test/Target/gemm16x16.mlir -o /tmp/vtaout \
  --emit-data --input mlir-vta/test/golden/input_16x16.bin \
  --weight mlir-vta/test/golden/weight_16x16.bin
# 用 Python 参考链路对同一 fixture 生成对照
cp mlir-vta/test/golden/input_16x16.bin mlir-vta/test/golden/weight_16x16.bin \
   standalone-vta/compiler_output/
cp standalone-vta/examples/vta_ir/matmul_16x16.json standalone-vta/compiler_output/
(cd standalone-vta/examples && python3 ../src/compiler/vta_compiler/main_vta_compiler.py \
   False True False ../config/vta_config.json ../compiler_output/matmul_16x16.json >/dev/null)
for f in input.bin weight.bin accumulator.bin out_init.bin metadata.csv memory_addresses.csv layers_name.csv; do
  cmp /tmp/vtaout/$f standalone-vta/compiler_output/$f && echo "$f OK"
done
```
Expected: 每个文件打印 `... OK`。

- [ ] **Step 4：提交**
```bash
git add mlir-vta && git commit -m "feat(mlir-vta): emit data bins and CSVs byte-exact"
```

---

### Task 5：FSIM 端到端验证

**Files:**
- Create: `mlir-vta/scripts/run_fsim.sh`

- [ ] **Step 1：写 `run_fsim.sh`（用 mlir-vta 产物喂 FSIM）**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
OUT=/tmp/vtaout
# 把 mlir-vta 产物拷进 standalone-vta/compiler_output（FSIM 固定从此读取）
cp $OUT/instructions.bin $OUT/uop.bin $OUT/input.bin $OUT/weight.bin \
   $OUT/accumulator.bin $OUT/out_init.bin $OUT/metadata.csv \
   $OUT/memory_addresses.csv $OUT/layers_name.csv $SVTA/compiler_output/
# add_accumulator.bin / expected_out_sram.bin：FSIM 单层需要，用 Python 产物或空文件补齐
: > $SVTA/compiler_output/add_accumulator.bin
cd $SVTA/examples
make fsim_compile_single_layer
make fsim_single_layer
tail -40 $SVTA/log_output/fsim_report.txt
```

- [ ] **Step 2：运行并对照结果矩阵**

Run: `chmod +x mlir-vta/scripts/run_fsim.sh && mlir-vta/scripts/run_fsim.sh`
Expected: `fsim_report.txt` 末尾打印 16×16 int32 结果矩阵；Profiler `gemm_counter:16, alu_counter:0`。与纯 Python 路径跑同一 fixture 的结果矩阵逐元素一致。

- [ ] **Step 3：提交**
```bash
git add mlir-vta/scripts && git commit -m "test(mlir-vta): FSIM end-to-end runner for 16x16 GEMM"
```

---

### Task 6：高层 `vta.gemm` + `lower-vta-gemm` 单块展开 pass

**Files:**
- Modify: `VTAOps.td`（加高层 `vta.gemm`）
- Create: `mlir-vta/lib/Transforms/LowerVTAGemm.cpp` + Pass 注册
- Modify: `vta-opt.cpp`（注册 pass）
- Test: `mlir-vta/test/Target/lower_gemm.mlir`

> 目的：证明「高层算子 → 低层指令」的 lowering 模式可行。第一阶段只支持**单块 16×16、无溢出**，直接生成 D.1 的固定 11 条（依赖位/factor 写死为黄金值）。通用化留给阶段二/三。

- [ ] **Step 1：加高层 op**

```tablegen
def VTA_GemmOp : VTA_Op<"gemm"> {
  let summary = "single 16x16 block GEMM: C = A * B (+ ACC)";
  let arguments = (ins I64Attr:$m, I64Attr:$n, I64Attr:$k);  // 阶段一仅接受 16,16,16
  let assemblyFormat = "attr-dict";
}
```

- [ ] **Step 2：写 `LowerVTAGemm` pass**

匹配 `vta.gemm {m=16,n=16,k=16}`，在其位置用 builder 依次创建 D.1 的 `vta.uop_table` + 11 条低层 op（属性照抄黄金值），然后 erase 原 op。对非 16×16 输入发 `emitError` 并标记 pass 失败（阶段一不支持）。

- [ ] **Step 3：注册 pass 到 `vta-opt`**

提供 `-lower-vta-gemm`。

- [ ] **Step 4：端到端断言（高层 → 二进制）**

`test/Target/lower_gemm.mlir`:
```mlir
// RUN: vta-opt -lower-vta-gemm %s -o %t.mlir
module { "vta.gemm"() {m=16:i64, n=16:i64, k=16:i64} : () -> () }
```
Run:
```bash
~/mlir-vta-build/bin/vta-opt -lower-vta-gemm mlir-vta/test/Target/lower_gemm.mlir -o /tmp/lowered.mlir
~/mlir-vta-build/bin/vta-translate /tmp/lowered.mlir -o /tmp/vtaout2
cmp /tmp/vtaout2/instructions.bin mlir-vta/test/golden/instructions.bin && echo "E2E INSN OK"
cmp /tmp/vtaout2/uop.bin mlir-vta/test/golden/uop.bin && echo "E2E UOP OK"
```
Expected: `E2E INSN OK` + `E2E UOP OK`。

- [ ] **Step 5：提交**
```bash
git add mlir-vta && git commit -m "feat(mlir-vta): high-level vta.gemm + single-block lowering pass"
```

---

## H. 自检清单（写完代码后逐项核对）

1. **Spec 覆盖**：B 节五项目标，每项都有对应 Task（1=骨架/dialect，2=低层 op，3=指令发射，4=数据/CSV，5=FSIM，6=高层 gemm+lowering）。✅
2. **字段一致性**：`VTAOps.td` 的属性名与 F 节 bit 区间、D.1 字段名（`dst_factor_out` 等）三处必须同名同义；发射器 `pack*` 读取的属性名与 td 定义一致。
3. **无占位符**：所有代码步骤含真实代码；CSV/hex 用 D 节实测值。
4. **ABI**：统一 `clang++`（=clang-13）构建，避免与 LLVM 13 的 libstdc++ ABI 冲突。
5. **磁盘**：build 目录在 `~`，不污染 `/mnt/c`。

---

## I. 已知风险与缓解

| 风险 | 缓解 |
|------|------|
| LLVM 13 API 与新版差异（`MlirOptMain` 签名、`parseSourceFile` 模板等） | 以 `/usr/local/llvm/include/mlir` 实际头文件为准；本计划 API 均为 13 可用形态，若签名不符按头文件微调 |
| `I1Attr`/`DefaultValuedAttr` 在 13 的写法差异 | 若 `I1Attr` 不可用，改用 `BoolAttr`/`UnitAttr`；round-trip 测试会暴露 |
| FSIM 单层还需 `add_accumulator.bin`/`expected_out_sram.bin` | Task 5 用空文件或从 Python 产物补齐；FSIM 单层对未用通道通常容忍空文件 |
| `expected_out_sram` 与 `inp@wgt` 不直接相等（实测） | 阶段一 TSIM 不做数值断言；FSIM 结果对照纯 Python 路径即可 |
| 阶段二的依赖信号量自动推导是真正难点 | 不在本计划；第一阶段用黄金固定依赖位，把发射器与 lowering 模式先验证扎实 |

---

## 执行交接

计划已保存至 `docs/plans/2026-06-02-mlir-vta-phase1-gemm.md`。两种执行方式：

1. **子代理驱动（推荐）**：每个 Task 派发一个全新子代理，任务间我来评审，迭代快。
2. **本会话内执行**：在当前会话按 Task 顺序执行，带检查点评审。

你想用哪种？
