# MLIR-VTA 阶段四实现计划 · ONNX 前端（增量 A：单层 QLinearConv）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `qlinearconv_debug.onnx` 立起 ONNX → MLIR → 二进制 → FSIM 的首条阶段四链路，并自动生成 `dependency.csv`。

**Architecture:** Python 桥接（`onnx` 包，无需 onnxruntime）解析 QLinearConv，复用 upstream im2col/ker2col，pad 到 16 倍数维，emit `linalg.matmul` MLIR；现有 `vta-opt`/`vta-translate` 管道 lower 到 `vtaisa`；`emit_dependency_csv.py` 写 FSIM 所需 CSV。

**Tech Stack:** Python 3 + onnx + numpy；MLIR/LLVM 13；standalone-vta FSIM。

> **实施状态：** 增量 A 已落地（2026-06）。`run_onnx_qconv.sh` 字节级 INSN/UOP + FSIM 对齐黄金。

---

## 文件结构

| 文件 | 动作 | 职责 |
|------|------|------|
| `docs/specs/phase4-onnx-frontend-design.md` | 创建 | 阶段四设计规格 |
| `docs/plans/phase4-onnx-frontend.md` | 创建 | 本计划 |
| `scripts/onnx_qconv_bridge.py` | 创建 | ONNX QLinearConv → bin + MLIR |
| `scripts/emit_dependency_csv.py` | 创建 | 层元数据 → dependency.csv |
| `scripts/run_onnx_qconv.sh` | 创建 | 端到端验收 |
| `scripts/make_golden_onnx_qconv.sh` | 创建 | 上游黄金生成 |
| `test/Target/qlinearconv_padded.mlir` | 创建 | 32×32×16 入口（桥接输出形态） |
| `docs/DESIGN_cn.md` | 修改 | §2.1 阶段四状态 |
| `docs/README.md` | 修改 | 索引 |
| `README.md` | 修改 | 当前状态 |

---

### Task 1：dependency.csv 发射器

**Files:**
- Create: `scripts/emit_dependency_csv.py`

- [x] **Step 1：** 实现 `write_dependency_csv(path, layers, image_rows, image_cols, output_layer, out_nchw)`，CRLF 行尾，格式对齐 `dependency_csv详解.md` §3–6。
- [x] **Step 2：** 支持单层 VTA GEMM（`processor=vta`, `reshape=im2row`）。

---

### Task 2：ONNX QLinearConv 桥接

**Files:**
- Create: `scripts/onnx_qconv_bridge.py`

- [x] **Step 1：** 用 `onnx` 解析模型，提取 QLinearConv 属性（kernel/stride/pads）、权重/偏置/scale/zp。
- [x] **Step 2：** im2col（复用 upstream 逻辑）+ ker2col，固定 seed=42 生成输入。
- [x] **Step 3：** 按 §4.2 padding 规则 pad 到 32×32 / 32×16；扩展偏置到 M×N。
- [x] **Step 4：** 写 raw bin + emit `test/Target/qlinearconv_padded.mlir` 形态 MLIR。
- [x] **Step 5：** 返回 `node_info` dict 供 CSV 发射。

---

### Task 3：端到端脚本

**Files:**
- Create: `scripts/run_onnx_qconv.sh`
- Create: `scripts/make_golden_onnx_qconv.sh`
- Create: `test/Target/qlinearconv_padded.mlir`

- [x] **Step 1：** `run_onnx_qconv.sh`：桥接 → bufferize → convert-linalg-to-vta → lower-vta-gemm(strategy=2) → translate → FSIM。
- [x] **Step 2：** FSIM 结果 vs 黄金，打印 ACCEPT/REJECT。
- [x] **Step 3：** `make_golden_onnx_qconv.sh` 用上游 VTA 编译器生成 `test/golden/qlinearconv_padded/`。

---

### Task 4：文档更新

- [x] 更新 `DESIGN_cn.md` §2.1 阶段四 →「增量 A 进行中」
- [x] 更新 `docs/README.md`、`README.md`

---

## 后续增量

| 增量 | 状态 |
|------|------|
| B–C, E | ✅ 已落地 |
| **F → G** | 🔄 见 [`phase4-increment-FG.md`](phase4-increment-FG.md) |
| D（onnx-mlir） | ⏸ 暂缓 |
