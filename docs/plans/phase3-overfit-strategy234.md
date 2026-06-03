# MLIR-VTA 阶段三增量：Overfit Strategy-2/3/4 多步调度

> **For agentic workers:** 按 task 顺序实施；每个 task 末尾有**字节级验证门**。Steps 用 `- [ ]` 跟踪。
>
> **必需子技能：** 使用 `superpowers:executing-plans` 逐任务实现此计划。
>
> **实施状态：✅ Strategy-2/3/4 已落地并通过全部验证门（2026-06）。** Task 0–5 完成：`vta.gemm` 加 `strategy` 可选属性（默认 1），`lower-vta-gemm` 扩展 Strategy-2（区域分块 192×16×192, 2步, 145 UOP, 159 指令）、Strategy-3（列主序 2064×16×16, 2步, 130 UOP, 144 指令）、Strategy-4（行主序 16×16×2048, 1步, 129 UOP, 138 指令）；三策略字节级验收（`cmp instructions.bin + uop.bin PASS`）；Strategy-1 overfit + 16×16 + 32×32 回归均通过。

**Goal:** 把 `lower-vta-gemm` 从「仅 strategy-1」扩展到 **strategy-2/3/4** 三种额外溢出调度策略，通过给 `vta.gemm` 加可选 `{strategy=N}` 属性选择策略，字节级对齐上游黄金（各策略有独立黄金目录）。

**架构：** `vta.gemm` 增加可选 `strategy` 整数属性（默认 1）；`lower-vta-gemm` 在 `isOverfit` 时根据该属性调用对应策略分支，各策略共享相同指令骨架（LOAD UOP reset → GEMM reset → 逐步 LOAD INP/WGT/ACC/UOP/GEMM → 逐块 STORE → NOP + FINISH），差异仅在 tile 参数和 UOP 索引计算。

**设计依据：**
- 上游 `src/compiler/vta_compiler/matrix_partitioning/gemm_strategies.py::strategy_2/3/4`
- 黄金：`test/golden/matmul_strategy{2,3,4}_*/`（已预生成）
- 现有实现：`lib/Transforms/LowerVTAGemm.cpp`（strategy-1 已落地）

---

## 关键数学——三种策略的 tile 参数

### SRAM 容量（默认 config）
```
INP_BUF = 128 块, WGT_BUF = 1024 块, ACC_BUF = OUT_BUF = 128 块
buf_size = min(128, 1024, 128) = 128
```

### Strategy-2（区域分块）
把 C 矩阵分成 tile_h × tile_w 大小的矩形区块，K 维按 tile_k 分段：
```cpp
int64_t tile_h = (int64_t)std::sqrt((double)ACC_BUF);
while (tile_h > 0 && ACC_BUF % tile_h != 0) tile_h--;
int64_t tile_w = ACC_BUF / tile_h;
tile_h = std::min(tile_h, Mb);
tile_w = std::min(tile_w, Nb);
int64_t tile_k = std::min({Kb, (tile_h > 0 ? INP_BUF / tile_h : Kb),
                               (tile_w > 0 ? WGT_BUF / tile_w : Kb)});
// M=192,K=16,N=192: tile_h=8, tile_w=12 (12*16/16=12), tile_k=1
```
每步 UOP（在 tile 内，共 tile_h * tile_w 条）：
- `dst = local_c_idx * 16`（tile 内行主序 C 块索引 × 16）
- `src = (local_c_idx / tile_w) * 16`（tile 内 A 行索引 × 16）
- `wgt = local_c_idx % tile_w`（tile 内 B 列索引）

每步 LOAD INP：`y = tile_h * tile_k`（tile 内 A 块数）；WGT：`y = tile_k * tile_w`；ACC：`y = tile_h * tile_w`。

### Strategy-3（列主序，delta 行）
按 B 列逐一，在每列内加载 delta 个 A/C 行：
```cpp
int64_t delta3 = std::min(bufSize, Mb);  // delta 个 C 行
int64_t nbDelta3 = Mb / delta3;
int64_t rem3 = Mb % delta3;
// M=2064,K=16,N=16: delta3=128, nbDelta3=16, rem3=1
```
每步（对 j 列, k A-col, delta 行）UOP（delta 条）：
- `dst = local_i * 16`（SRAM 中 C 行偏移 × 16，行主序）
- `src = local_i * 16`（SRAM 中 A 行偏移 × 16）
- `wgt = 0`（单 B 块，SRAM 偏移 0）

每步 LOAD INP：`y = delta`（delta 个 A 行块）；WGT：`y = 1`（单 B 块）；ACC：`y = delta`（delta 个 C 行块，首次 k=0 才 LOAD）。

### Strategy-4（行主序，delta 列）
按 A 行逐一，在每行内加载 delta 个 B/C 列：
```cpp
int64_t delta4 = std::min(bufSize, Nb);  // delta 个 C 列
int64_t nbDelta4 = Nb / delta4;
int64_t rem4 = Nb % delta4;
// M=16,K=16,N=2048: delta4=128, nbDelta4=16, rem4=0
```
每步（对 i 行, k A-col, delta 列）UOP（delta 条）：
- `dst = local_j * 16`（SRAM 中 C 列偏移 × 16）
- `src = 0`（单 A 块，SRAM 偏移 0）
- `wgt = local_j`（SRAM 中 B 列偏移）

每步 LOAD INP：`y = 1`（单 A 块）；WGT：`y = delta`（delta 个 B 列块）；ACC：`y = delta`（delta 个 C 列块，首次 k=0 才 LOAD）。

---

## 黄金文件（已预生成）

| 策略 | 维度 | 步数 | UOP数 | 指令数 | 黄金目录 |
|------|------|------|-------|--------|---------|
| Strategy-2 | M=192, K=16, N=192 (tile_h=8, tile_w=12, tile_k=1) | 2 | 145 | 159 | `test/golden/matmul_strategy2_192x16x192/` |
| Strategy-3 | M=2064, K=16, N=16 (delta=128 行, 2步) | 2 | 130 | 144 | `test/golden/matmul_strategy3_2064x16x16/` |
| Strategy-4 | M=16, K=16, N=2048 (delta=128 列, 1步) | 1 | 129 | 138 | `test/golden/matmul_strategy4_16x16x2048/` |

黄金中已含 `instructions.bin`, `uop.bin`, `memory_addresses.csv`, `metadata.csv`。

---

## Task 0：给 `vta.gemm` 加可选 `strategy` 属性

**Files:**
- Modify: `include/mlir-vta/Dialect/VTA/VTAOps.td`
- Modify: `lib/Dialect/VTA/VTAOps.cpp`（verifier）

- [x] **Step 1：VTAOps.td 加 `strategy` 属性**

  在 `GemmOp` 定义中加入可选整数属性（默认 1，即 strategy-1）：

  ```tablegen
  // 在 GemmOp 的 arguments 中现有字段后加：
  DefaultValuedAttr<I64Attr, "1">:$strategy
  ```

  找到 `GemmOp` 的 `arguments` 定义，在 `$name` 之后（或 `$acc` 之后）加入该属性。保持 `assemblyFormat` 中可选打印：
  ```tablegen
  // assemblyFormat 末尾加 attr-dict 覆盖或者手动打印，
  // 最简单：在现有 attr-dict 打印里 strategy 会自动出现
  ```

- [x] **Step 2：verifier 检查 strategy 范围**

  在 `lib/Dialect/VTA/VTAOps.cpp` 的 `GemmOp::verify()` 中加入：
  ```cpp
  if (strategy() < 1 || strategy() > 4) {
    return emitOpError("strategy must be in [1,4], got ") << strategy();
  }
  ```

- [x] **Step 3：构建验证**
  ```bash
  touch /mnt/c/MLIR-VTA/mlir-vta/include/mlir-vta/Dialect/VTA/VTAOps.td
  cmake --build ~/mlir-vta-build -j"$(nproc)" --target vta-opt 2>&1 | tail -5
  ```
  Expected: build success.

- [x] **Step 4：round-trip 测试**
  ```bash
  printf 'func @m(%%a: memref<32x32xi32>, %%b: memref<32x32xi32>, %%c: memref<32x32xi32>) {\n  vta.gemm ins(%%a, %%b : memref<32x32xi32>, memref<32x32xi32>) outs(%%c : memref<32x32xi32>) {strategy = 1 : i64}\n  return\n}\n' > /tmp/gemm_s1.mlir
  ~/mlir-vta-build/bin/vta-opt /tmp/gemm_s1.mlir | grep -q "vta.gemm" && echo "strategy attr ROUNDTRIP OK"
  ```

- [x] **Step 5：提交**
  ```bash
  printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' 'git add -A' 'git commit -m "feat(mlir-vta): add optional strategy attr to vta.gemm (default=1)"' > /tmp/do_commit.sh && bash /tmp/do_commit.sh
  ```

---

## Task 1：实现 Strategy-2（区域分块）

**Files:**
- Modify: `lib/Transforms/LowerVTAGemm.cpp`
- Create: `test/golden/matmul_strategy2_192x16x192/`（已预生成）

- [x] **Step 1：在 `isOverfit` 分支中读取 `strategy` 属性**

  在 `LowerVTAGemm.cpp` 的 overfit 分支开头加入策略读取：
  ```cpp
  const int64_t strategyId = g.strategy();  // 1/2/3/4
  ```
  将现有 strategy-1 代码移入 `if (strategyId == 1) { ... }` 块。

- [x] **Step 2：计算 strategy-2 tile 参数**

  ```cpp
  } else if (strategyId == 2) {
    // tile_h × tile_w 区域分块，tile_k K 维分段
    int64_t tile_h = (int64_t)std::sqrt((double)kAccBuf);
    while (tile_h > 0 && kAccBuf % tile_h != 0) tile_h--;
    if (tile_h == 0) tile_h = 1;
    int64_t tile_w = kAccBuf / tile_h;
    tile_h = std::min(tile_h, Mb);
    tile_w = std::min(tile_w, Nb);
    int64_t tile_k = Kb;
    if (tile_h > 0) tile_k = std::min(tile_k, kInpBuf / tile_h);
    if (tile_w > 0) tile_k = std::min(tile_k, kWgtBuf / tile_w);
    if (tile_k == 0) tile_k = 1;
    ...
  ```

- [x] **Step 3：构建 strategy-2 的 UOP 表和指令序列**

  Strategy-2 UOP（每 C tile 共 tile_h * tile_w 条）：
  ```cpp
  // 清空并重建 UOP 表（仅在 isOverfit 分支重建）
  uopDst.clear(); uopSrc.clear(); uopWgt.clear();
  uopDst.push_back(0); uopSrc.push_back(0); uopWgt.push_back(0); // reset

  // Strategy-2: 对每个 (i_tile, j_tile) 的 C tile，对每个 k_tile:
  for (int64_t i = 0; i < Mb; i += tile_h) {
    int64_t cur_h = std::min(tile_h, Mb - i);
    for (int64_t j = 0; j < Nb; j += tile_w) {
      int64_t cur_w = std::min(tile_w, Nb - j);
      for (int64_t k = 0; k < Kb; k += tile_k) {
        int64_t cur_k = std::min(tile_k, Kb - k);
        // UOP entries for this tile step: cur_h * cur_w gemm ops
        for (int64_t r = 0; r < cur_h * cur_w; ++r) {
          int64_t local_i = r / cur_w;  // row within tile
          int64_t local_j = r % cur_w;  // col within tile
          // dst: C block at (local_i, local_j) in SRAM
          // For each k in cur_k, we need one UOP per (r):
          // but the GEMM loops over k internally? No - strategy-2 uses get_gemm_operations
          // which produces one uop per (a, b) pair where a picks k blocks.
          // Let's check: cur_k A blocks (rows of A tile) × cur_w B blocks (cols of B tile)
          // = cur_h * cur_w gemm uops per (i_tile, j_tile, k_tile)?
          // Actually: each step produces cur_h * cur_w * cur_k GeMM ops
          // (one per (c_local, a_col_local) pair)
          // Wait - let me re-read strategy_2 Python code more carefully
        }
      }
    }
  }
  ```

  > **重要：** 需要先用 Python 验证 strategy-2 的 UOP 结构（每步有多少条 UOP，索引规律）再写 C++ 代码。

- [x] **Step 4：验证 UOP 结构（在写 C++ 前先用 Python 验证）**

  ```python
  # 在 standalone-vta 环境中运行
  import sys
  sys.path.insert(0, 'src/compiler/vta_compiler')
  from matrix_partitioning.gemm_strategies import strategy_2
  from matrix_partitioning.utils_strategies import get_gemm_operations

  Mb, Kb, Nb = 12, 1, 12  # 192x16x192 in blocks
  steps = strategy_2(nb_A=Mb*Kb, A_blocks_col=Kb, nb_B=Kb*Nb, B_blocks_col=Nb,
                     nb_X=Mb*Nb, nb_C=Mb*Nb, C_blocks_col=Nb,
                     inp_block_buffer_size=128, wgt_block_buffer_size=1024,
                     acc_block_buffer_size=128, out_block_buffer_size=128)
  print(f'nb_steps={len(steps)}')
  for si, step in enumerate(steps):
      load_A, load_B, load_X, mem, _, store_C, ops = step
      print(f'Step {si}: A={load_A[:3]}... B={load_B[:3]}... X={load_X[:3]}... store={store_C[:3]}...')
      print(f'  GeMM ops ({len(ops)}): first={ops[:3]}')
  ```

  运行此脚本确认步骤数和每步的 ops 列表，从而推导 UOP 规律。

- [x] **Step 5：根据 Python 验证结果实现 C++ 代码（具体代码在完成 Step 4 后补充）**

  关键结论（需 Step 4 验证后填写）：
  - 每步的 GeMM ops 数 = ?
  - UOP `dst/src/wgt` 索引规律 = ?
  - 每步 LOAD INP y_size = ?, WGT y_size = ?, ACC y_size = ?

- [x] **Step 6：构建**
  ```bash
  touch /mnt/c/MLIR-VTA/mlir-vta/lib/Transforms/LowerVTAGemm.cpp
  cmake --build ~/mlir-vta-build -j"$(nproc)" --target vta-opt vta-translate 2>&1 | tail -5
  ```

- [x] **Step 7：字节级验证门**
  ```bash
  GD=/mnt/c/MLIR-VTA/mlir-vta/test/golden/matmul_strategy2_192x16x192
  printf 'func @main(%%a: memref<192x16xi32>, %%b: memref<16x192xi32>, %%c: memref<192x192xi32>) {\n  vta.gemm ins(%%a, %%b : memref<192x16xi32>, memref<16x192xi32>) outs(%%c : memref<192x192xi32>) {strategy = 2 : i64}\n  return\n}\n' > /tmp/gs2.mlir
  rm -rf /tmp/os2 && mkdir -p /tmp/os2
  ~/mlir-vta-build/bin/vta-opt -lower-vta-gemm /tmp/gs2.mlir -o /tmp/low_s2.mlir
  ~/mlir-vta-build/bin/vta-translate /tmp/low_s2.mlir -o /tmp/os2
  cmp /tmp/os2/instructions.bin $GD/instructions.bin && echo "S2 INSN OK (159 insns)"
  cmp /tmp/os2/uop.bin          $GD/uop.bin          && echo "S2 UOP OK (145 uops)"
  ```

- [x] **Step 8：回归 strategy-1 + CASE1（字节级不变）**
  ```bash
  # strategy-1 overfit 回归
  cd /mnt/c/MLIR-VTA/mlir-vta && GD1=test/golden/matmul_overfit_16x2064x16
  printf 'func @main(%%a: memref<16x2064xi32>, %%b: memref<2064x16xi32>, %%c: memref<16x16xi32>) {\n  vta.gemm ins(%%a, %%b : memref<16x2064xi32>, memref<2064x16xi32>) outs(%%c : memref<16x16xi32>)\n  return\n}\n' > /tmp/gs1.mlir
  rm -rf /tmp/os1 && mkdir -p /tmp/os1
  ~/mlir-vta-build/bin/vta-opt -lower-vta-gemm /tmp/gs1.mlir -o /tmp/low_s1.mlir
  ~/mlir-vta-build/bin/vta-translate /tmp/low_s1.mlir -o /tmp/os1
  cmp /tmp/os1/instructions.bin $GD1/instructions.bin && echo "S1 INSN PASS"
  cmp /tmp/os1/uop.bin          $GD1/uop.bin          && echo "S1 UOP PASS"
  # 16x16 CASE1 回归
  rm -rf /tmp/reg16c && mkdir -p /tmp/reg16c
  ~/mlir-vta-build/bin/vta-opt -lower-vta-gemm test/Target/lower_gemm.mlir -o /tmp/reg16c/low.mlir
  ~/mlir-vta-build/bin/vta-translate /tmp/reg16c/low.mlir -o /tmp/reg16c
  cmp /tmp/reg16c/instructions.bin test/golden/instructions.bin && echo "16x16 INSN PASS"
  ```

- [x] **Step 9：提交**
  ```bash
  printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' 'git add -A' 'git commit -m "feat(mlir-vta): overfit strategy-2 region-based tiling (192x16x192, 2-step, 145 uop)"' > /tmp/do_commit.sh && bash /tmp/do_commit.sh
  ```

---

## Task 2：实现 Strategy-3（列主序，delta 行）

**Files:**
- Modify: `lib/Transforms/LowerVTAGemm.cpp`

- [x] **Step 1：计算 strategy-3 参数**

  Strategy-3：对每个 B 列 j，加载 delta 个 A/C 行（SRAM 复用），K 维完整遍历。
  ```cpp
  } else if (strategyId == 3) {
    const int64_t delta3 = std::min(bufSize, Mb);  // delta C rows per step
    const int64_t nbDelta3 = Mb / delta3;
    const int64_t rem3 = Mb % delta3;
    ...
  ```

- [x] **Step 2：用 Python 验证 strategy-3 UOP 结构（先验证再写 C++）**

  ```python
  from matrix_partitioning.gemm_strategies import strategy_3
  Mb, Kb, Nb = 129, 1, 1  # M=2064, K=16, N=16 in blocks
  steps = strategy_3(nb_A=Mb*Kb, A_blocks_col=Kb, nb_B=Kb*Nb, B_blocks_col=Nb,
                     nb_X=Mb*Nb, nb_C=Mb*Nb, C_blocks_col=Nb,
                     inp_block_buffer_size=128, wgt_block_buffer_size=1024,
                     acc_block_buffer_size=128, out_block_buffer_size=128)
  print(f'nb_steps={len(steps)}')
  for si, step in enumerate(steps):
      load_A, load_B, load_X, mem, _, store_C, ops = step
      print(f'Step {si}: A={load_A[:5]} B={load_B} X={load_X[:5]} store={store_C[:5]}')
      print(f'  ops[:3]={ops[:3]}')
  ```

  从输出确认：
  - 每步 delta 个 A 块 × 1 个 B 块 → delta 条 UOP（局部 SRAM 索引）
  - `dst = local_i * 16`（C 行在 SRAM 中的偏移）
  - `src = local_i * 16`（A 行在 SRAM 中的偏移）
  - `wgt = 0`（B 块总是 SRAM 偏移 0）

- [x] **Step 3：实现 strategy-3 指令发射**

  Strategy-3 的 UOP 表（所有 j × K × delta_step 的 UOP 顺序追加）：
  ```cpp
  // 对每个 delta_step（nbDelta3 + (rem3>0 ? 1 : 0) 组）：
  //   对每个 j in [0, Nb)：
  //     对每个 k in [0, Kb)：
  //       追加 cur_delta 条 UOP：dst=local_i*16, src=local_i*16, wgt=0
  for (int64_t dStep = 0; dStep <= nbDelta3; ++dStep) {
    int64_t cur_delta = (dStep < nbDelta3) ? delta3 : rem3;
    if (cur_delta == 0) continue;
    for (int64_t j = 0; j < Nb; ++j) {
      for (int64_t k = 0; k < Kb; ++k) {
        for (int64_t localI = 0; localI < cur_delta; ++localI) {
          uopDst.push_back(localI * 16);
          uopSrc.push_back(localI * 16);
          uopWgt.push_back(0);
        }
      }
    }
  }
  ```

  指令序列（对每个 step/j/k 三层循环，每次发射 INP+WGT+[ACC]+UOP+GEMM+[STORE]）：
  - LOAD INP y=cur_delta（delta 个 A 行块，从全局 A 块偏移读）
  - LOAD WGT y=1（1 个 B 列块）
  - LOAD ACC y=cur_delta（仅 k==0 时），从全局 ACC 偏移
  - LOAD UOP（cur_delta 条 uop）
  - GEMM push_prev=1，push_next=1 仅最后步（k==Kb-1 且是最后 j 且是最后 delta_step）
  - STORE（仅 k==Kb-1 时，逐 C 块 STORE）

- [x] **Step 4：字节级验证门**
  ```bash
  GD=/mnt/c/MLIR-VTA/mlir-vta/test/golden/matmul_strategy3_2064x16x16
  printf 'func @main(%%a: memref<2064x16xi32>, %%b: memref<16x16xi32>, %%c: memref<2064x16xi32>) {\n  vta.gemm ins(%%a, %%b : memref<2064x16xi32>, memref<16x16xi32>) outs(%%c : memref<2064x16xi32>) {strategy = 3 : i64}\n  return\n}\n' > /tmp/gs3.mlir
  rm -rf /tmp/os3 && mkdir -p /tmp/os3
  ~/mlir-vta-build/bin/vta-opt -lower-vta-gemm /tmp/gs3.mlir -o /tmp/low_s3.mlir
  ~/mlir-vta-build/bin/vta-translate /tmp/low_s3.mlir -o /tmp/os3
  cmp /tmp/os3/instructions.bin $GD/instructions.bin && echo "S3 INSN OK (144 insns)"
  cmp /tmp/os3/uop.bin          $GD/uop.bin          && echo "S3 UOP OK (130 uops)"
  ```

- [x] **Step 5：回归（同 Task 1 Step 8）**

- [x] **Step 6：提交**
  ```bash
  printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' 'git add -A' 'git commit -m "feat(mlir-vta): overfit strategy-3 column-major (2064x16x16, 2-step, 130 uop)"' > /tmp/do_commit.sh && bash /tmp/do_commit.sh
  ```

---

## Task 3：实现 Strategy-4（行主序，delta 列）

**Files:**
- Modify: `lib/Transforms/LowerVTAGemm.cpp`

- [x] **Step 1：计算 strategy-4 参数**

  Strategy-4：对每个 A 行 i，加载 delta 个 B/C 列（SRAM 复用），K 维完整遍历。
  ```cpp
  } else if (strategyId == 4) {
    const int64_t delta4 = std::min(bufSize, Nb);  // delta C cols per step
    const int64_t nbDelta4 = Nb / delta4;
    const int64_t rem4 = Nb % delta4;
    ...
  ```

- [x] **Step 2：用 Python 验证 strategy-4 UOP 结构**

  ```python
  from matrix_partitioning.gemm_strategies import strategy_4
  Mb, Kb, Nb = 1, 1, 128  # M=16, K=16, N=2048 in blocks
  steps = strategy_4(nb_A=Mb*Kb, A_blocks_col=Kb, nb_B=Kb*Nb, B_blocks_col=Nb,
                     nb_X=Mb*Nb, nb_C=Mb*Nb, C_blocks_col=Nb,
                     inp_block_buffer_size=128, wgt_block_buffer_size=1024,
                     acc_block_buffer_size=128, out_block_buffer_size=128)
  print(f'nb_steps={len(steps)}')
  for si, step in enumerate(steps):
      load_A, load_B, load_X, mem, _, store_C, ops = step
      print(f'Step {si}: A={load_A} B={load_B[:5]} X={load_X[:5]} store={store_C[:5]}')
      print(f'  ops[:3]={ops[:3]}')
  ```

  预期确认：
  - 每步 1 个 A 块 × delta 个 B 块 → delta 条 UOP
  - `dst = local_j * 16`（C 列在 SRAM 中的偏移）
  - `src = 0`（A 块总是 SRAM 偏移 0）
  - `wgt = local_j`（B 列在 SRAM 中的偏移）

- [x] **Step 3：实现 strategy-4 指令发射**

  UOP 表构建（对 i × delta_step × k 三层，每步 cur_delta 条）：
  ```cpp
  for (int64_t i = 0; i < Mb; ++i) {
    for (int64_t dStep = 0; dStep <= nbDelta4; ++dStep) {
      int64_t cur_delta = (dStep < nbDelta4) ? delta4 : rem4;
      if (cur_delta == 0) continue;
      for (int64_t k = 0; k < Kb; ++k) {
        for (int64_t localJ = 0; localJ < cur_delta; ++localJ) {
          uopDst.push_back(localJ * 16);
          uopSrc.push_back(0);
          uopWgt.push_back(localJ);
        }
      }
    }
  }
  ```

  指令序列（i × delta_step × k 三层）：
  - LOAD INP y=1（单 A 行块）
  - LOAD WGT y=cur_delta（delta 个 B 列块）
  - LOAD ACC y=cur_delta（仅 k==0 时）
  - LOAD UOP（cur_delta 条 uop）
  - GEMM push_prev=1，push_next=1 仅最后步
  - STORE（仅 k==Kb-1 时，逐 C 块 STORE）

- [x] **Step 4：字节级验证门**
  ```bash
  GD=/mnt/c/MLIR-VTA/mlir-vta/test/golden/matmul_strategy4_16x16x2048
  printf 'func @main(%%a: memref<16x16xi32>, %%b: memref<16x2048xi32>, %%c: memref<16x2048xi32>) {\n  vta.gemm ins(%%a, %%b : memref<16x16xi32>, memref<16x2048xi32>) outs(%%c : memref<16x2048xi32>) {strategy = 4 : i64}\n  return\n}\n' > /tmp/gs4.mlir
  rm -rf /tmp/os4 && mkdir -p /tmp/os4
  ~/mlir-vta-build/bin/vta-opt -lower-vta-gemm /tmp/gs4.mlir -o /tmp/low_s4.mlir
  ~/mlir-vta-build/bin/vta-translate /tmp/low_s4.mlir -o /tmp/os4
  cmp /tmp/os4/instructions.bin $GD/instructions.bin && echo "S4 INSN OK (138 insns)"
  cmp /tmp/os4/uop.bin          $GD/uop.bin          && echo "S4 UOP OK (129 uops)"
  ```

- [x] **Step 5：回归**

- [x] **Step 6：提交**
  ```bash
  printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' 'git add -A' 'git commit -m "feat(mlir-vta): overfit strategy-4 row-major (16x16x2048, 1-step, 129 uop)"' > /tmp/do_commit.sh && bash /tmp/do_commit.sh
  ```

---

## Task 4：保存原始数据 + FSIM 黄金

**Files:**
- Create: `test/golden/matmul_strategy{2,3,4}_*/`（追加 FSIM 黄金）

- [x] **Step 1：为三个策略生成并保存输入数据黄金**

  ```bash
  SVTA=/mnt/c/MLIR-VTA/standalone-vta
  for case in strategy2_192x16x192 strategy3_2064x16x16 strategy4_16x16x2048; do
    GD=/mnt/c/MLIR-VTA/mlir-vta/test/golden/matmul_${case}
    json="matmul_${case}"
    cd $SVTA/examples
    python3 ../src/compiler/vta_compiler/main_vta_compiler.py True True False \
        ../config/vta_config.json ../compiler_output/${json}.json 2>/dev/null
    cp ../compiler_output/input*.bin ../compiler_output/weight*.bin \
       ../compiler_output/accumulator.bin ../compiler_output/out_init.bin \
       "$GD/" 2>/dev/null || true
    echo "Data golden saved for $case"
  done
  ```

- [x] **Step 2：运行 FSIM 为三个策略生成结果矩阵黄金**

  对每个策略（用已有 MLIR 生成的 instructions.bin + 上游数据），先把 MLIR 产出的 instructions.bin 暂存，然后用**上游 Python 编译器**生成 FSIM 黄金：

  ```bash
  SVTA=/mnt/c/MLIR-VTA/standalone-vta
  for case in strategy2_192x16x192 strategy3_2064x16x16 strategy4_16x16x2048; do
    GD=/mnt/c/MLIR-VTA/mlir-vta/test/golden/matmul_${case}
    json="matmul_${case}"
    cd $SVTA/examples
    # 重新生成上游黄金（已含 input/weight/accumulator/out_init）
    python3 ../src/compiler/vta_compiler/main_vta_compiler.py True True False \
        ../config/vta_config.json ../compiler_output/${json}.json 2>/dev/null
    cp ../compiler_output/layers_name.csv "$GD/" 2>/dev/null || true
    : > $SVTA/compiler_output/add_accumulator.bin
    rm -f $SVTA/log_output/fsim_report.txt
    make fsim_compile_single_layer > /dev/null 2>&1
    make fsim_single_layer > /dev/null 2>&1
    sed -n '/Final result/,/^} $/p' $SVTA/log_output/fsim_report.txt \
        > "$GD/fsim_result.txt"
    echo "FSIM golden saved for $case: $(wc -l < $GD/fsim_result.txt) lines"
  done
  ```

- [x] **Step 3：提交黄金数据**
  ```bash
  printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' 'git add -A' 'git commit -m "test(mlir-vta): add data and FSIM goldens for strategy-2/3/4"' > /tmp/do_commit.sh && bash /tmp/do_commit.sh
  ```

---

## Task 5：更新文档

- [x] `DESIGN_cn.md` §2.1 阶段三状态更新：加 strategy-2/3/4 已落地
- [x] `docs/plans/phase3-generalized-gemm.md` 状态更新
- [x] `docs/README.md` 索引加 `phase3-overfit-strategy234.md`
- [x] `README.md` 当前状态表 + 尚未实现更新
- [x] 提交 `docs(mlir-vta): document overfit strategy-2/3/4`

---

## 自检清单

1. **字节级验证门**：Task 1/2/3 各有 cmp 门，对 instructions.bin + uop.bin
2. **回归**：Task 1/2/3 均验证 strategy-1 overfit + CASE1 16×16/32×32 不变
3. **黄金预生成**：Task 4 补充 FSIM 黄金（字节级门已足够，FSIM 为额外信号）
4. **strategy 属性**：Task 0 加 `strategy` 属性，不破坏无属性的旧用例（默认=1）

---

## 附录：先验——Python 快速验证脚本

在开始 C++ 编码前，先在 standalone-vta 环境运行以下脚本确认 UOP 规律：

```bash
cd /mnt/c/MLIR-VTA/standalone-vta
python3 -c "
import sys
sys.path.insert(0, 'src/compiler/vta_compiler')
from matrix_partitioning.gemm_strategies import strategy_2, strategy_3, strategy_4

print('=== Strategy 2 (Mb=12,Kb=1,Nb=12) ===')
steps = strategy_2(144,1,144,12, 144,144,12, 128,1024,128,128)
print(f'steps={len(steps)}')
for si,(A,B,X,m,t,C,ops) in enumerate(steps):
    print(f'  step{si}: A={A[:3]}.. B={B[:3]}.. X={X[:3]}.. store={C[:3]}.. ops={len(ops)} first_op={ops[0] if ops else None}')

print()
print('=== Strategy 3 (Mb=129,Kb=1,Nb=1) ===')
steps = strategy_3(129,1,1,1, 129,129,1, 128,1024,128,128)
print(f'steps={len(steps)}')
for si,(A,B,X,m,t,C,ops) in enumerate(steps[:3]):
    print(f'  step{si}: A={A[:3]}.. B={B} X={X[:3]}.. store={C[:3]}.. ops={len(ops)} first_op={ops[0] if ops else None}')

print()
print('=== Strategy 4 (Mb=1,Kb=1,Nb=128) ===')
steps = strategy_4(1,1,128,128, 128,128,128, 128,1024,128,128)
print(f'steps={len(steps)}')
for si,(A,B,X,m,t,C,ops) in enumerate(steps[:3]):
    print(f'  step{si}: A={A} B={B[:3]}.. X={X[:3]}.. store={C[:3]}.. ops={len(ops)} first_op={ops[0] if ops else None}')
" 2>&1
```
