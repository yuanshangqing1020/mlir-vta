# MLIR-VTA 阶段三增量：真·层间串联（`fsim_nn` 2×16×16 GEMM）

> **实施状态：✅ 已落地（2026-06）。** 两层 16×16 GEMM 经 `fsim_nn` 串联执行，256/256 元素与 `fsim_single_layer` 独立验证完全一致。脚本 `scripts/run_fsim_nn_2layer.sh` 可重复运行端到端验证。

---

## 背景与意义

前述"两层 16×16 多层 GEMM"（`-vta-dram-allocation` + 字节级验收）中，上游 `fsim_single_layer` **不支持层间数据传递**——每层独立从磁盘读 `input*.bin`，无法把 L0 的输出当 L1 的输入。

`fsim_nn` 是上游 `standalone-vta` 中专为整网推理设计的仿真器：按 `dependency.csv` 调度，把前一层的 `ctx.res`（int8）链式传给下一层的 `inpA`，真正实现层间数据流。

**本增量目标**：手写 `dependency.csv`，让 `fsim_nn` 运行 2×16×16 GEMM 的串联（NNA→NNB），验证数值正确性。这是阶段四 NN 编译器集成之前的简化版层间串联验证。

---

## 关键设计发现

### 1. 连续 DRAM 地址是必要条件

`fsim_nn` 同时预加载所有层到 VTA 虚拟 DRAM。每层的 `instructions.bin` 里的 `dram_base` 地址是编译时确定的。如果分别单独编译两层（每层从地址 0 开始），两层的地址空间会重叠，VTA 执行结果全为 0。

**解决方案**：一次调用上游编译器编译两层（`main_vta_compiler.py ... L0.json L1.json`）。编译器会把 L1 的起始地址设为 L0 末尾地址（如 L0 结束于 `0x6fff`，L1 从 `0x7000` 开始）。

### 2. `dependency.csv` 中 GEMM 层的 im2row 参数

对 16×16 矩阵乘（A[16,16] × B[16,16]），映射为"1×1 卷积"参数：

```csv
NNA,vta,im2row,0,1,0,1,0,1,0,1,16,1,16,1,1,1,1,0,0,0,0,16,1,16,0,1,1,INP,1,image
NNB,vta,im2row,0,1,0,1,0,1,0,1,16,1,16,1,1,1,1,0,0,0,0,16,1,16,0,1,1,INP,1,NNA
```

- `image,16,16`：输入展平为 16行（=H×W）× 16列（=channel）
- `tensor_channel=16, H=1, W=16, kh=1, kw=1, sh=1, pad=0`
- 1×1 卷积时 im2row 等于恒等映射，inpA = A 矩阵

### 3. `input_nn.bin` 必须是 int8 格式

`fsim_nn` 读 `input_nn.bin` 为 `int8`，经 `data_formatting(int8_vec, 16, 16, 16, True)` 后再 `reshape` 为 int32 的 `inpA`。与 `fsim_single_layer` 用 `inputNNA.bin`（int32）不同。

---

## 验证结果

```
=== FSIM_NN 2-LAYER GEMM CHAIN: ACCEPT (256/256 elements match) ===
fsim_nn (NNA→NNB serial) == fsim_single_layer(NNB with NNA output)
```

数据：seed=42，A/W0/W1 均为 16×16 随机矩阵（值域 [-2, 2]），保证 VTA GEMM 中间结果在 int8 范围内（无截断）。

---

## 文件与脚本

| 文件 | 说明 |
|------|------|
| `scripts/run_fsim_nn_2layer.sh` | 端到端验证脚本（生成数据→编译→运行 fsim_nn→独立验证→ACCEPT） |
| 上游 `standalone-vta/src/simulators/functional_simulator/src/fsim_nn.cc` | 整网串联仿真器（未修改） |
| 上游 `standalone-vta/src/compiler/vta_compiler/main_vta_compiler.py` | 支持多 IR 连续地址编译（argv[5:]） |

---

## 局限说明

- **简化版**：仅 VTA GEMM 层，无量化（scale=1, offset=0）、无 CPU 算子（qadd/concat 等）。完整的量化 NN 推理依赖 NN 编译器（`vta_backend.py`）生成 `dependency.csv`，属阶段四范畴。
- **手写 dependency.csv**：im2row 参数需手动对应 GEMM 的张量形状；真实网络由 NN 编译器自动生成。
- **地址布局**：需一次编译多层（连续地址）；mlir-vta 的 `-vta-dram-allocation` pass 生成的地址与上游一致，可直接复用。
