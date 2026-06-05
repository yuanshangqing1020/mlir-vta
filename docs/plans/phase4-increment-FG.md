# MLIR-VTA 阶段四增量 F→G 实施计划

> **For agentic workers:** 按 Task 顺序执行；F 全部验收通过后再启动 G。增量 D（onnx-mlir）**不在本计划范围**——待本地 onnx-mlir 环境就绪后单独开计划。

**Goal:** 在 Python ONNX 桥接路径上补齐 **CPU 回退算子**（F），并完成 **LeNet-5 整网** `fsim_nn` 数值端到端（G）。

**Architecture:** CPU 层（`qadd`/`concat`）**不需要** MLIR lowering——`fsim_nn` 已内置 CPU 实现；mlir-vta 只需正确发射 `dependency.csv` + VTA 层二进制。G 在 F 基础上叠更多 VTA/CPU 层，最终 `final_output.bin` vs `reference.bin`。

**Tech Stack:** Python 3 + onnx + numpy；现有 `vta-opt`/`vta-translate`；standalone-vta `fsim_nn` + `reference_onnx.py`。

---

## 背景

| 项 | 说明 |
|----|------|
| FSIM CPU 路径 | `fsim_nn.cc` 已实现 `processor == "qadd"` / `"concat"`（读两路/多路 `ctx.res`，量化合并） |
| mlir-vta 缺口 | `emit_dependency_csv.py` 对双输入层的 `offsetU/V` 写死为 0；无 QLinearAdd/Concat ONNX 桥接脚本 |
| LeNet-5 | upstream 有 `insn_lenet5_layer*.py`、`examples_compute/lenet5/`；整网 ONNX 需生成或从 pattern 裁剪 |
| 增量 D | onnx-mlir 替换 Python 桥接——**暂缓**，不阻塞 F/G |

---

## 增量 F：CPU 算子 + 混合调度

### F.1 验收模型（由简到繁）

| 里程碑 | 模型 | 说明 |
|--------|------|------|
| **F-a** | `qadd_two_mul.onnx` | 同输入两路 `QLinearMul`（VTA `mul_constant`）→ `QLinearAdd`（CPU `qadd`） |
| **F-b** | `qconcat_two_branch.onnx` | 两路 Conv 输出通道维 `QLinearConcat`（CPU `concat`，axis=1） |
| **F-c** | `pattern_debug.onnx` 子集 | 对齐 upstream `dependency_csv详解.md` §7 五层 pattern（可选） |

### F.2 交付物

| 文件 | 职责 |
|------|------|
| `scripts/emit_dependency_csv.py` | 支持 `processor=qadd|concat`、双/多父层 `offsetU/V/scaleU/V`、父层名列 |
| `scripts/gen_qadd_two_mul.py` | 生成 F-a ONNX（seed=42，ir_version=9） |
| `scripts/onnx_qadd_two_mul_bridge.py` | 解析 ONNX → 2×VTA gemm 层 + 1×qadd 层元数据 + MLIR |
| `scripts/make_golden_qadd_two_mul.sh` | NumPy/ORT 参考 → `reference.bin` + `input_nn.bin` |
| `scripts/run_onnx_qadd_two_mul.sh` | 编译 VTA 层 → staging → `fsim_nn` → `cmp reference.bin` |
| `test/golden/qadd_two_mul/` | golden fixtures |

Concat（F-b）同理：`gen_*` / `onnx_*_bridge.py` / `run_onnx_qconcat.sh`。

### F.3 Task 清单

#### Task F-1：`emit_dependency_csv` 泛化

- [x] 层 dict 可选 `offsetU/scaleU/offsetV/scaleV`
- [x] `parent_layers` 长度 2+ 时写 `nb_inp` 与父层名列
- [x] `processor=qadd` + `reshape=int32` 已用于 F-a

#### Task F-2：QLinearAdd 双分支（F-a）

- [x] `gen_qadd_two_mul.py` / `onnx_qadd_two_mul_bridge.py` / `run_onnx_qadd_two_mul.sh`
- [x] `fsim_nn` 3 步调度（2 VTA + 1 `qadd`）可跑通并写出 `final_output.bin`
- [x] **数值 golden 对齐**（`fsim_reference_utils.py` + NCHW 参考；`cmp` 通过）

#### Task F-3：QLinearConcat（F-b）

- [x] `qconcat_two_branch.onnx`：两路 Conv + Concat
- [x] `processor=concat`，`reshape=False`（**必须** False，勿用 `im2row` 否则 fsim segfault）
- [x] 数值 golden + `run_onnx_qconcat.sh`（`ACCEPT`）

---

## 增量 G：LeNet-5 整网

### G.1 范围与分步

LeNet-5 典型层：Conv → ReLU → AvgPool → …（5 层卷积块 + FC）。**分两步降低风险：**

| 里程碑 | 范围 | 验收 |
|--------|------|------|
| **G-a** | Layer 1 only（Conv+ReLU+AvgPool，tutorial2 对齐） | 单层/三层 VTA + CPU pool？AvgPool 可能需 CPU 或 alu |
| **G-b** | `lenet5_mini.onnx`：2×（Conv+ReLU）链，5×5 与 `conv_relu_debug` 同尺度 | `run_onnx_lenet5.sh` → `ACCEPT` |

> **说明：** 完整 LeNet-5（AvgPool/FC/28×28）仍待后续；16×16 空间下 NumPy QLinearConv 与 VTA fsim 存在系统性数值差（见 `two_qlinearconv`），故 G 采用 5×5 可比对 golden。

### G.2 交付物

| 文件 | 职责 |
|------|------|
| `scripts/gen_lenet5_mini.py` | 2×Conv+ReLU 链 ONNX（5×5） |
| `scripts/onnx_lenet5_mini_bridge.py` | 4 层 VTA（2×gemm + 2×alu）+ dependency |
| `scripts/gen_lenet5_mini_reference.py` | NumPy golden |
| `scripts/run_onnx_lenet5.sh` | 端到端 fsim_nn + `cmp` |
| `test/golden/lenet5_mini/` | reference / input_nn |

### G.3 Task 清单

#### Task G-3：lenet5_mini（G-b，已完成）

- [x] `gen_lenet5_mini.py` / `onnx_lenet5_mini_bridge.py` / `run_onnx_lenet5.sh`
- [x] 4 步 fsim_nn（2×Conv + 2×ReLU）+ 数值 `cmp` 通过
- [ ] 完整 LeNet-5（AvgPool/FC）——后续里程碑

---

## 增量 D（暂缓，仅记录）

| 条件 | 动作 |
|------|------|
| 本地安装 onnx-mlir + LLVM 对齐版本 | 新建 `phase4-increment-D.md` |
| spike | `QLinearConv` → `linalg` → 接入现有 `convert-linalg-to-vta` |
| 切换策略 | Python 桥接与 onnx-mlir 路径 **并行** 验收，golden 不变 |

---

## 验证命令（目标态）

```bash
# F-a
bash scripts/run_onnx_qadd_two_mul.sh

# F-b
bash scripts/run_onnx_qconcat.sh

# G-b
bash scripts/run_onnx_lenet5.sh
```

---

## 参考

- [`specs/phase4-onnx-frontend-design.md`](../specs/phase4-onnx-frontend-design.md)
- [`standalone-vta/docs/dependency_csv详解.md`](../../../standalone-vta/docs/dependency_csv详解.md) §7（qadd 五层 pattern）
- [`standalone-vta/src/simulators/functional_simulator/src/fsim_nn.cc`](../../../standalone-vta/src/simulators/functional_simulator/src/fsim_nn.cc)（`qadd`/`concat` 分支）
