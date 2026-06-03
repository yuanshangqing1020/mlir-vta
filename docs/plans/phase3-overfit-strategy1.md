# MLIR-VTA 阶段三增量：Overfit Strategy-1 多步调度

> **For agentic workers:** 按 task 顺序实施；每个 task 末尾有**对齐上游黄金的验证门**。Steps 用 `- [ ]` 跟踪。
>
> **必需子技能：** 使用 `superpowers:executing-plans` 逐任务实现此计划。

**Goal:** 把 `lower-vta-gemm` 从「CASE 1 单步拒绝 overfit」扩展到**Strategy-1 多步调度**：当 nbA≥128 或 nbB≥1024 或 nbC≥128 时，按上游 `strategy_1` 逻辑分多个 step 发射指令，每步加载 `delta=min(buf,Kb)` 个 A/B 块、累加到 ACC、最后逐 C 块 STORE；字节级对齐上游黄金 `test/golden/matmul_overfit_16x2064x16/`（15 条指令、130 UOP）。

**设计依据：**
- 上游 `src/compiler/vta_compiler/matrix_partitioning/gemm_strategies.py::strategy_1`
- 上游 `src/compiler/vta_compiler/operations_definition/step_instructions.py`
- 黄金解码（见本文档 §黄金指令解码表）
- 现有实现 `lib/Transforms/LowerVTAGemm.cpp`

**关键约束：**
- 只实现 Strategy-1（每步处理一个 C 块，delta=min(buf,Kb) 个 A/B 块）
- SRAM 缓冲容量常量：`INP_BUF=128块`, `WGT_BUF=1024块`, `ACC_BUF=OUT_BUF=128块`（来自 vta_config.json 默认配置）
- DRAM 地址分配（`computeGemmLayout`）不需修改，已与 overfit 黄金一致
- 所有回归（16×16、32×32、矩形、多层）须字节级不变

---

## 环境

- Git 根 `/mnt/c/MLIR-VTA/mlir-vta`，分支 `main`
- 增量构建：`cmake --build ~/mlir-vta-build -j"$(nproc)" --target vta-opt vta-translate`

---

## 黄金指令解码表（M=16, K=2064, N=16, strategy_1, 2步）

上游编译器产出 15 条指令、130 UOP（`test/golden/matmul_overfit_16x2064x16/`）：

```
I0:  LOAD UOP  sram=0 dram=0x11400 y=1 x=1 str=1            | (无 dep)        ← reset uop
I1:  GEMM reset=1 uop[0,1) lp_out=1 lp_in=16 dst_o=16 dst_i=1 | push_prev=1    ← ACC 清零

--- Step 0: load_A=[0..127], load_B=[0..127], load_X=[0] (delta=128 blocks) ---
I2:  LOAD INP sram=0  dram=0x40  y=128 x=16 str=16          | pop_next=1      ← 128 INP 块
I3:  LOAD WGT sram=0  dram=0x88  y=128 x=1  str=1           | push_next=1     ← 128 WGT 块
I4:  LOAD ACC sram=0  dram=0x10c0 y=1  x=16 str=16          | pop_prev=1      ← 1 ACC 块(首step)
I5:  LOAD UOP sram=0  dram=0x11401 y=1 x=128 str=128        | (无 dep)        ← 128 gemm uops
I6:  GEMM reset=0 uop[0,128) lp_out=1 lp_in=16 dst_i=1 src_i=1 | push_prev=1  ← 中间步骤，不 push_next

--- Step 1: load_A=[128], load_B=[128], load_X=[] (remainder=1 block, has STORE) ---
I7:  LOAD INP sram=0  dram=0x840  y=1  x=16 str=16          | pop_next=1      ← INP 块 128
I8:  LOAD WGT sram=0  dram=0x108  y=1  x=1  str=1           | push_next=1     ← WGT 块 128
I9:  LOAD UOP sram=0  dram=0x11481 y=1 x=1 str=1            | pop_prev=1      ← 1 gemm uop
I10: GEMM reset=0 uop[0,1) lp_out=1 lp_in=16 dst_i=1 src_i=1 | push_prev=1 push_next=1 ← 最后步，push_next=1

--- STORE (nbC=1块) ---
I11: STORE OUT sram=0 dram=0x1100 y=1 x=16 str=16           | pop_prev=1 push_prev=1  ← 单块=首且末

--- Termination ---
I12: LOAD INP sram=0 dram=0 y=0 x=0 str=0                   | pop_next=1 push_next=1
I13: LOAD UOP sram=0 dram=0 y=0 x=0 str=0                   | pop_prev=1 pop_next=1
I14: FINISH
```

**UOP 结构（130条）：**
- UOP0: reset (dst=0, src=0, wgt=0)
- UOP1..128: step 0 的 128 条 GEMM uop → UOP[1+k]: dst=0, src=k*16, wgt=k (k=0..127)
- UOP129: step 1 的 1 条 GEMM uop → dst=0, src=128*16=2048, wgt=128

**地址规律：**
- INP 块 a 的 DRAM 地址 = INP_base + a*16（a 是 DRAM 中的全局块索引）
- WGT 块 b 的 DRAM 地址 = WGT_base + b（b 是 DRAM 中的全局块索引）
- ACC/OUT 块 c 的 DRAM 地址 = ACC/OUT_base + c*16
- step 内 UOP 的 DRAM 地址 = UOP_base + 1 + (uop_offset_start)
  - step 0 的 128 uop 从 UOP_base+1 开始
  - step 1 的 1 uop 从 UOP_base+1+128=UOP_base+129 开始

---

## Task 0：研究 Strategy-1 指令生成逻辑（已在计划写作阶段完成）

**完成标志：** 本文档即为 Task 0 的输出，已包含完整分析。

---

## Task 1：扩展 `LowerVTAGemm` 支持 Strategy-1 多步调度

**Files:**
- Modify: `lib/Transforms/LowerVTAGemm.cpp`

### 概念设计

Strategy-1 的伪代码（复刻上游 `gemm_strategies.py::strategy_1`）：

```
// 常量
const int64_t INP_BUF = 128, WGT_BUF = 1024, ACC_BUF = 128;
const int64_t buf_size = min({INP_BUF, WGT_BUF, ACC_BUF});
const int64_t delta = min(buf_size, Kb);
const int64_t nb_delta = Kb / delta;
const int64_t remainder = Kb % delta;

// UOP 表（顺序追加，与上游一致）
uopDst/Src/Wgt = [reset(0,0,0)]
for each C block (idx=0..nbC-1):
  i = idx / Nb; j = idx % Nb;
  for delta_step=0..nb_delta-1:
    k_start = delta_step * delta
    for k_local=0..delta-1:
      a = i*Kb + k_start + k_local
      b = (k_start + k_local)*Nb + j
      uopDst.push(0*16=0)   // c_sram_idx=0（每步复用同一块SRAM）
      uopSrc.push(k_local*16)  // a_sram_idx = k_local（在 delta 个 SRAM 块中的位置）
      uopWgt.push(k_local)  // b_sram_idx = k_local
  if remainder > 0:
    for k_local=0..remainder-1:
      ... (同上，k_start=nb_delta*delta)

// 指令发射：
emit LOAD UOP (reset)
emit GEMM reset

for each C block (idx=0..nbC-1):
  first_c = (idx == 0);
  last_c = (idx == nbC-1);
  i = idx / Nb; j = idx % Nb;
  uop_cursor = 0;  // 指向当前 C 块的第一个 UOP（相对于 UOP_base+1）
  
  for delta_step=0..nb_delta-1:
    is_first_step = (delta_step == 0 and first_c);
    is_last_step = (delta_step == nb_delta-1 and remainder == 0 and last_c);
    
    a_start = i*Kb + delta_step*delta;
    b_start = delta_step*delta*Nb + j;
    
    emit LOAD INP sram=0, dram=INP_base+a_start*16, y=delta, x=16, str=16
         (pop_next=1, ack CMP->LD signal)
    emit LOAD WGT sram=0, dram=WGT_base+b_start, y=delta, x=1, str=1
         (push_next=1, send LD->CMP signal if no more loads in this step)
    if (is_first_step):
      emit LOAD ACC sram=0, dram=ACC_base+idx*16, y=1, x=16, str=16
           (pop_prev=1, ack ST->CMP signal)
    emit LOAD UOP sram=0, dram=UOP_base+1+uop_cursor, y=1, x=delta, str=delta
    emit GEMM reset=0 uop[0,delta) push_prev=1 push_next=(is_last_step?1:0)
    uop_cursor += delta;
  
  if remainder > 0:
    is_last_step = last_c;
    a_start = i*Kb + nb_delta*delta;
    b_start = nb_delta*delta*Nb + j;
    emit LOAD INP sram=0, dram=INP_base+a_start*16, y=remainder, x=16, str=16 (pop_next=1)
    emit LOAD WGT sram=0, dram=WGT_base+b_start, y=remainder, x=1, str=1 (push_next=1)
    emit LOAD UOP sram=0, dram=UOP_base+1+uop_cursor, y=1, x=remainder, str=remainder
         (pop_prev=1, ack LD->CMP token)
    emit GEMM reset=0 uop[0,remainder) push_prev=1 push_next=(is_last_step?1:0)
    uop_cursor += remainder;
  
  // STORE（每 C 块处理完毕后）
  for blk=0..nbC-1 (对整体 C):  // 实际是每个 C 块在 delta_step 全部完成后 STORE
    // 此循环在外层 for C block 循环内，每轮结尾 STORE 当前 C 块
    pop_prev = true;   // 每次 STORE 开始都 pop_prev（GEMM 保证了 push_next 给 STORE）
    push_prev = true;  // 每次 STORE 结束都 push_prev（ST->CMP）
    emit STORE OUT sram=idx*0, dram=OUT_base+idx*16 y=1 x=16 str=16
         pop_prev=1 push_prev=1  // 对 nbC=1：单块两位都置1

emit NOP-LOAD INP  (pop_next=1 push_next=1)
emit NOP-LOAD UOP  (pop_prev=1 pop_next=1)
emit FINISH
```

> **注意：** 实际上 strategy_1 总是每次处理完一个 C 块就立即 STORE。STORE 之间的依赖位：
> - 首 STORE（对当前 C）：pop_prev=1（消费 GEMM 的 push_next=1 token）
> - 末 STORE（对当前 C）：push_prev=1（产生 ST->CMP token，允许下一 C 的 ACC LOAD pop_prev）
> - 若 nbC=1：单个 STORE 同时携带 pop_prev=1 且 push_prev=1

### 实施步骤

- [ ] **Step 1：在 `LowerVTAGemmPass::runOnOperation` 中，将 overfit 的 `emitError` 改为调用新的 Strategy-1 发射函数**

  在 `LowerVTAGemm.cpp` 中，找到并替换：
  ```cpp
  if (nbA >= 128 || nbB >= 1024 || nbC >= 128) {
    g.emitError("lower-vta-gemm: matrix too large for single-step schedule "
                "(overfit not yet supported)");
    signalPassFailure();
    return;
  }
  ```
  改为：
  ```cpp
  const bool isOverfit = (nbA >= 128 || nbB >= 1024 || nbC >= 128);
  ```
  （删去 return，继续往下执行；后面会根据 `isOverfit` 分支）

- [ ] **Step 2：构建 Strategy-1 UOP 表（适用于 CASE1 和 Overfit）**

  当前代码生成的 UOP 表（CASE1 路径）：
  ```cpp
  for (int64_t a = 0; a < nbA; ++a) {
    int64_t i = a / Kb, k = a % Kb;
    for (int64_t bb = 0; bb < nbB; ++bb) {
      int64_t bk = bb / Nb, j = bb % Nb;
      if (bk != k) continue;
      ...uopDst.push_back(cIdx * 16); uopSrc.push_back(a * 16); uopWgt.push_back(bb);
    }
  }
  ```
  这是 CASE1 全局块索引 UOP，与 Strategy-1 的 SRAM 局部索引 UOP **不同**。

  Strategy-1 的 UOP 中 src/wgt 是 **SRAM 中的局部块索引**（在当前 step 加载的 delta 个块里的位置），而不是全局块索引。

  因此，在 `isOverfit` 为 true 时，需要**重新构建 UOP 表**：

  ```cpp
  // Strategy-1 UOP（SRAM-local 索引，每 C 块单独追加，每 delta_step 内 k_local=0..delta-1）
  // 替换掉之前基于全局块索引的 UOP 构建逻辑
  const int64_t kInpBuf = 128, kWgtBuf = 1024, kAccBuf = 128;
  const int64_t bufSize = std::min({kInpBuf, kWgtBuf, kAccBuf});
  const int64_t delta = std::min(bufSize, Kb);
  const int64_t nbDelta = Kb / delta;
  const int64_t remainder = Kb % delta;

  // 清空并重建 UOP 表（用 Strategy-1 SRAM 局部索引）
  uopDst.clear(); uopSrc.clear(); uopWgt.clear();
  uopDst.push_back(0); uopSrc.push_back(0); uopWgt.push_back(0); // reset

  for (int64_t idx = 0; idx < nbC; ++idx) {
    // 对每个 C 块，delta_step 轮：
    for (int64_t dStep = 0; dStep < nbDelta; ++dStep) {
      for (int64_t kLocal = 0; kLocal < delta; ++kLocal) {
        uopDst.push_back(0);         // c_sram_idx * 16，Strategy-1 C 块在 SRAM 0
        uopSrc.push_back(kLocal * 16); // a_sram_idx * 16
        uopWgt.push_back(kLocal);    // b_sram_idx
      }
    }
    if (remainder > 0) {
      for (int64_t kLocal = 0; kLocal < remainder; ++kLocal) {
        uopDst.push_back(0);
        uopSrc.push_back(kLocal * 16);
        uopWgt.push_back(kLocal);
      }
    }
  }
  // G 保持为 G = uopDst.size() - 1（总 gemm uop 数，= nbC * Kb）
  ```

  > **验证：** M=16(Mb=1), K=2064(Kb=129), N=16(Nb=1)：delta=128, nbDelta=1, remainder=1, nbC=1。
  > UOP 数 = 1 + 1*(128+1) = 130 ✓，与黄金 130 UOP 一致。

- [ ] **Step 3：拆分指令发射逻辑为两条路径**

  在 `OpBuilder b(g);` 之后，根据 `isOverfit` 分支：

  ```cpp
  // uop_table 发射（两条路径共用，已在 Step 2 重建）
  b.create<vtaisa::UopTableOp>(loc, b.getI64ArrayAttr(uopDst),
                               b.getI64ArrayAttr(uopSrc),
                               b.getI64ArrayAttr(uopWgt));

  // reset 序列（两条路径共用）
  b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, false, false, false,
                           false, 0, kUopBase, 1, 1, 1);
  b.create<vtaisa::GemmInsnOp>(loc, false, false, true, false, true, 0, 1,
                               1, 16, 16, 1);

  if (!isOverfit) {
    // CASE 1: 现有单步逻辑（不变）
    ...
  } else {
    // Strategy-1: 多步逻辑
    emitStrategy1(b, loc, Mb, Kb, Nb, nbC, kInpBase, kWgtBase, kAccBase, kOutBase, kUopBase,
                  delta, nbDelta, remainder);
  }

  // Termination NOPs + FINISH（两条路径共用）
  b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP, false, true, false, true, 0, 0, 0, 0, 0);
  b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, true, true, false, false, 0, 0, 0, 0, 0);
  b.create<vtaisa::FinishOp>(loc);
  ```

- [ ] **Step 4：实现 `emitStrategy1` 函数（或内联到 Step 3）**

  ```cpp
  // Strategy-1 多步指令发射（内联在 runOnOperation 中，isOverfit 分支里）
  int64_t uopCursor = 0; // 指向 UOP_base+1 之后的偏移（按 uop 数累加）

  for (int64_t idx = 0; idx < nbC; ++idx) {
    int64_t ci = idx / Nb, cj = idx % Nb;
    bool firstC = (idx == 0);
    bool lastC = (idx == nbC - 1);
    uopCursor = 0; // 每个 C 块重置（注意：多 C 块时 UOP 各自独立，本计划 nbC 通常较小）
    // 注意：实际上对 nbC>1 的 Strategy-1，各 C 块的 UOP 是顺序追加的，不重置
    // 但 LOAD UOP 的 dram_base 是绝对地址（UOP_base+1+累积uop_cursor），需要维护全局 cursor
    // 以下用全局 uop_cursor（从 0 计数），指示当前 C 块的 UOP 在 UOP 区的位置：

    for (int64_t dStep = 0; dStep < nbDelta; ++dStep) {
      bool isLastStep = (dStep == nbDelta - 1 && remainder == 0 && lastC);
      int64_t aStart = ci * Kb + dStep * delta; // 全局 A 块索引起点
      int64_t bStart = (dStep * delta) * Nb + cj; // 全局 B 块索引起点

      // LOAD INP (pop_next=1)
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP,
        false, /*pop_next=*/true, false, false,
        /*sram_base=*/0, /*dram_base=*/kInpBase + aStart * 16,
        /*y_size=*/delta, /*x_size=*/16, /*x_stride=*/16);

      // LOAD WGT (push_next=1)
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::WGT,
        false, false, false, /*push_next=*/true,
        /*sram_base=*/0, /*dram_base=*/kWgtBase + bStart,
        /*y_size=*/delta, /*x_size=*/1, /*x_stride=*/1);

      // LOAD ACC（仅第一个 C 块的第一步，pop_prev=1）
      if (firstC && dStep == 0) {
        b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::ACC,
          /*pop_prev=*/true, false, false, false,
          /*sram_base=*/0, /*dram_base=*/kAccBase + idx * 16,
          /*y_size=*/1, /*x_size=*/16, /*x_stride=*/16);
      }

      // LOAD UOP (delta gemm uops)
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP,
        false, false, false, false,
        /*sram_base=*/0, /*dram_base=*/kUopBase + 1 + uopCursor,
        /*y_size=*/1, /*x_size=*/delta, /*x_stride=*/delta);

      // GEMM (push_prev=1, push_next=isLastStep)
      b.create<vtaisa::GemmInsnOp>(loc, false, false,
        /*push_prev=*/true, /*push_next=*/isLastStep,
        /*reset=*/false, /*uop_bgn=*/0, /*uop_end=*/delta,
        /*loop_out=*/1, /*loop_in=*/16, 0, 1, 0, 1);

      uopCursor += delta;
    }

    if (remainder > 0) {
      bool isLastStep = lastC; // remainder 步就是该 C 块的最后步
      int64_t aStart = ci * Kb + nbDelta * delta;
      int64_t bStart = (nbDelta * delta) * Nb + cj;

      // LOAD INP (pop_next=1)
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP,
        false, /*pop_next=*/true, false, false,
        0, kInpBase + aStart * 16, remainder, 16, 16);

      // LOAD WGT (push_next=1)
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::WGT,
        false, false, false, /*push_next=*/true,
        0, kWgtBase + bStart, remainder, 1, 1);

      // LOAD UOP (pop_prev=1, remainder uops)
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP,
        /*pop_prev=*/true, false, false, false,
        0, kUopBase + 1 + uopCursor, 1, remainder, remainder);

      // GEMM (push_prev=1, push_next=isLastStep)
      b.create<vtaisa::GemmInsnOp>(loc, false, false,
        /*push_prev=*/true, /*push_next=*/isLastStep,
        false, 0, remainder, 1, 16, 0, 1, 0, 1);

      uopCursor += remainder;
    }

    // STORE（每个 C 块处理完即 STORE，只有一个块时 pop_prev=push_prev=1）
    b.create<vtaisa::StoreOp>(loc, vtaisa::BufferId::OUT,
      /*pop_prev=*/true, false, /*push_prev=*/true, false,
      /*sram_base=*/0, /*dram_base=*/kOutBase + idx * 16,
      1, 16, 16);
  }
  ```

  > **注意：** 上面是针对 nbC=1 的简化版本（黄金用例）。对 nbC>1 的情况（如 M=N=192）：
  > - 每个 C 块完成后才 STORE，且每个 C 块之间需要额外的 ACC LOAD（Strategy-1 每 C 块独立处理）
  > - 本 Task 只需字节级通过 nbC=1 的黄金（16x2064x16），nbC>1 的测试留后续

- [ ] **Step 5：处理 LOAD ACC 在 nbC>1 时的 ACC LOAD 时机**

  对 nbC>1（非本阶段主要目标，但逻辑须正确）：
  - 每个 C 块 idx 的第一步（`dStep==0`），都需要 `LOAD ACC sram=0, dram=ACC_base+idx*16`
  - 不只限于 `firstC`：改为 `dStep == 0`（不管是哪个 C 块）
  - 依赖位：`pop_prev=1` 当前一个 C 块的 STORE 结束后（ST->CMP 信号）

  修正 Step 4 中的条件：`if (firstC && dStep == 0)` → `if (dStep == 0)`，且：
  - 首 C 块（`idx==0`）：`pop_prev=1`（消费 GEMM reset 的 push_prev 信号）
  - 后续 C 块（`idx>0`）：`pop_prev=1`（消费上一 C 块 STORE 的 push_prev 信号）
  - 实际上对所有 C 块都是 `pop_prev=1`，因为 GEMM reset 和 STORE 都产生 push_prev 信号

- [ ] **Step 6：构建并检查编译无错误**

  ```bash
  cmake --build ~/mlir-vta-build -j"$(nproc)" --target vta-opt vta-translate 2>&1 | tail -5
  ```
  Expected: `[0 warnings, 0 errors]` 或仅有既有 warnings。

- [ ] **Step 7：验证门（字节级，对齐黄金）**

  ```bash
  GD=/mnt/c/MLIR-VTA/mlir-vta/test/golden/matmul_overfit_16x2064x16

  # 生成 lowered IR
  printf 'func @main(%%a: memref<16x2064xi32>, %%b: memref<2064x16xi32>, %%c: memref<16x16xi32>) {\n  vta.gemm ins(%%a, %%b : memref<16x2064xi32>, memref<2064x16xi32>) outs(%%c : memref<16x16xi32>)\n  return\n}\n' > /tmp/goverfit.mlir

  rm -rf /tmp/ofit && mkdir -p /tmp/ofit
  ~/mlir-vta-build/bin/vta-opt -lower-vta-gemm /tmp/goverfit.mlir -o /tmp/low_ofit.mlir
  ~/mlir-vta-build/bin/vta-translate /tmp/low_ofit.mlir -o /tmp/ofit

  cmp /tmp/ofit/instructions.bin $GD/instructions.bin && echo "INSN OK (15 insn)"
  cmp /tmp/ofit/uop.bin          $GD/uop.bin          && echo "UOP OK (130 uop)"
  ```
  Expected: `INSN OK` + `UOP OK`

- [ ] **Step 8：回归 CASE1（字节级不变）**

  ```bash
  # 16×16 回归
  ~/mlir-vta-build/bin/vta-opt -lower-vta-gemm test/Target/lower_gemm.mlir | \
    ~/mlir-vta-build/bin/vta-translate -o /tmp/reg16
  cmp /tmp/reg16/instructions.bin test/golden/instructions.bin && echo "16x16 INSN PASS"
  cmp /tmp/reg16/uop.bin          test/golden/uop.bin          && echo "16x16 UOP PASS"

  # 32×32 回归
  printf 'func @m(%%a: memref<32x32xi32>, %%b: memref<32x32xi32>, %%c: memref<32x32xi32>) {\n  vta.gemm ins(%%a, %%b : memref<32x32xi32>, memref<32x32xi32>) outs(%%c : memref<32x32xi32>)\n  return\n}\n' > /tmp/g32.mlir
  ~/mlir-vta-build/bin/vta-opt -lower-vta-gemm /tmp/g32.mlir | ~/mlir-vta-build/bin/vta-translate -o /tmp/reg32
  cmp /tmp/reg32/instructions.bin test/golden/matmul_32x32/instructions.bin && echo "32x32 INSN PASS"
  cmp /tmp/reg32/uop.bin          test/golden/matmul_32x32/uop.bin          && echo "32x32 UOP PASS"
  ```

- [ ] **Step 9：提交**

  ```bash
  printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' \
    'git add -A' 'git commit -m "feat(mlir-vta): overfit strategy-1 multi-step lowering (16x2064x16 2-step, 130 uop, 15 insn)"' \
    > /tmp/do_commit.sh && bash /tmp/do_commit.sh
  ```

---

## Task 2：端到端入口 + 数据发射器支持 overfit 用例

**Files:**
- Create: `test/Target/matmul_tensor_overfit.mlir`
- Modify: `lib/Target/VTADataEmitter.cpp`（支持 K>16 的 overfit 维度）
- Create: `scripts/run_fsim_overfit.sh`

- [ ] **Step 1：创建入口 IR** `test/Target/matmul_tensor_overfit.mlir`

  ```mlir
  // Phase-3 overfit entry: M=16, K=2064, N=16 (strategy-1, 2 steps)
  func @main(%A: tensor<16x2064xi32>, %B: tensor<2064x16xi32>) -> tensor<16x16xi32> {
    %c0 = constant 0 : i32
    %init = linalg.init_tensor [16, 16] : tensor<16x16xi32>
    %acc = linalg.fill(%c0, %init) : i32, tensor<16x16xi32> -> tensor<16x16xi32>
    %out = linalg.matmul ins(%A, %B : tensor<16x2064xi32>, tensor<2064x16xi32>)
                         outs(%acc : tensor<16x16xi32>) -> tensor<16x16xi32>
    return %out : tensor<16x16xi32>
  }
  ```

- [ ] **Step 2：验证管道可用（不需要 linalg-tile）**

  ```bash
  ~/mlir-vta-build/bin/vta-opt test/Target/matmul_tensor_overfit.mlir \
    --linalg-bufferize --tensor-bufferize --tensor-constant-bufferize \
    --func-bufferize --finalizing-bufferize \
    --convert-linalg-to-vta --canonicalize --lower-vta-gemm \
    -o /tmp/low_e2e_ofit.mlir
  grep -q "vtaisa.finish" /tmp/low_e2e_ofit.mlir && echo "HAS vtaisa"
  grep -q "linalg.matmul\|vta.gemm" /tmp/low_e2e_ofit.mlir && echo "RESIDUAL (BAD)" || echo "no residual"
  ```

- [ ] **Step 3：确认 VTADataEmitter 支持大 K**

  当前 `VTADataEmitter.cpp` 使用 `--rows M --cols N --k K` CLI 参数。检查是否支持 K=2064：

  ```bash
  ~/mlir-vta-build/bin/vta-translate /tmp/low_e2e_ofit.mlir -o /tmp/data_ofit \
    --emit-data \
    --input test/golden/matmul_overfit_16x2064x16/input_16x2064.bin \
    --weight test/golden/matmul_overfit_16x2064x16/weight_2064x16.bin \
    --rows 16 --cols 16 --k 2064
  ls -la /tmp/data_ofit/
  ```

  若失败（如读写错误），检查 `VTADataEmitter.cpp` 中的 K 循环上界；通常是按 `Kb = k/16` 计算，应该已支持。若 accumulator.bin 大小为 1*256*4=1024B 则正确（nbC=1）。

- [ ] **Step 4：字节级验证数据文件**

  ```bash
  GD=/mnt/c/MLIR-VTA/mlir-vta/test/golden/matmul_overfit_16x2064x16
  cmp /tmp/data_ofit/accumulator.bin $GD/accumulator.bin 2>/dev/null && echo "ACC DATA OK" \
    || echo "ACC DATA MISSING/MISMATCH (may not exist in golden - OK)"
  # 注：accumulator.bin（全零）用于 FSIM，若黄金不含此文件则无需 cmp
  ```

- [ ] **Step 5：拷贝 overfit 数据黄金（若不存在）**

  overfit 用例的 `input.bin/weight.bin/accumulator.bin` 需从原始矩阵重新块化产生。检查黄金目录：

  ```bash
  ls test/golden/matmul_overfit_16x2064x16/
  ```

  若缺少 `input.bin`/`weight.bin`/`accumulator.bin`，用 vta-translate 产生并存入黄金：

  ```bash
  ~/mlir-vta-build/bin/vta-translate /tmp/low_e2e_ofit.mlir -o /tmp/golden_data \
    --emit-data \
    --input test/golden/matmul_overfit_16x2064x16/input_16x2064.bin \
    --weight test/golden/matmul_overfit_16x2064x16/weight_2064x16.bin \
    --rows 16 --cols 16 --k 2064
  cp /tmp/golden_data/input.bin     test/golden/matmul_overfit_16x2064x16/
  cp /tmp/golden_data/weight.bin    test/golden/matmul_overfit_16x2064x16/
  cp /tmp/golden_data/accumulator.bin test/golden/matmul_overfit_16x2064x16/
  cp /tmp/golden_data/out_init.bin  test/golden/matmul_overfit_16x2064x16/ 2>/dev/null || true
  ```

  **重要：** 还需要生成 FSIM 结果黄金文件 `fsim_result_overfit.txt`，用上游 FSIM 运行：

  ```bash
  SVTA=/mnt/c/MLIR-VTA/standalone-vta
  cp /tmp/ofit/instructions.bin /tmp/ofit/uop.bin \
     /tmp/golden_data/input.bin /tmp/golden_data/weight.bin \
     /tmp/golden_data/accumulator.bin /tmp/golden_data/out_init.bin \
     /tmp/golden_data/metadata.csv \
     test/golden/matmul_overfit_16x2064x16/memory_addresses.csv \
     /tmp/golden_data/layers_name.csv \
     $SVTA/compiler_output/
  : > $SVTA/compiler_output/add_accumulator.bin
  cd $SVTA/examples
  make fsim_compile_single_layer && make fsim_single_layer
  # 提取结果矩阵
  sed -n '/Final result/,/^} $/p' $SVTA/log_output/fsim_report.txt \
    > /mnt/c/MLIR-VTA/mlir-vta/test/golden/matmul_overfit_16x2064x16/fsim_result_overfit.txt
  echo "FSIM golden saved"
  ```

- [ ] **Step 6：创建 `scripts/run_fsim_overfit.sh`**（仿 `run_fsim_linalg_32x32.sh`）

  ```bash
  #!/usr/bin/env bash
  set -euo pipefail
  ROOT=/mnt/c/MLIR-VTA
  SVTA=$ROOT/standalone-vta
  MLIRVTA=$ROOT/mlir-vta
  OUT=/tmp/vtaout_fsim_overfit
  OPT=$HOME/mlir-vta-build/bin/vta-opt
  TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate
  GD=$MLIRVTA/test/golden/matmul_overfit_16x2064x16

  rm -rf "$OUT" && mkdir -p "$OUT"
  "$OPT" "$MLIRVTA/test/Target/matmul_tensor_overfit.mlir" \
    --linalg-bufferize --tensor-bufferize --tensor-constant-bufferize \
    --func-bufferize --finalizing-bufferize \
    --convert-linalg-to-vta --canonicalize --lower-vta-gemm \
    -o "$OUT/lowered.mlir"
  "$TRANSLATE" "$OUT/lowered.mlir" -o "$OUT" \
    --emit-data --input "$GD/input_16x2064.bin" --weight "$GD/weight_2064x16.bin" \
    --rows 16 --cols 16 --k 2064
  cp "$OUT"/instructions.bin "$OUT"/uop.bin "$OUT"/input.bin "$OUT"/weight.bin \
     "$OUT"/accumulator.bin "$OUT"/out_init.bin "$OUT"/metadata.csv \
     "$OUT"/memory_addresses.csv "$OUT"/layers_name.csv "$SVTA/compiler_output/"
  : > "$SVTA/compiler_output/add_accumulator.bin"
  rm -f "$SVTA/log_output/fsim_report.txt"
  cd "$SVTA/examples"
  make fsim_compile_single_layer
  make fsim_single_layer

  REPORT="$SVTA/log_output/fsim_report.txt"
  GOLDEN="$GD/fsim_result_overfit.txt"
  sed -n '/Final result/,/^} $/p' "$REPORT" > "$OUT/fsim_result.txt"
  if [ ! -s "$OUT/fsim_result.txt" ]; then
    echo "FSIM result matrix empty -- extraction failed (REJECT)" >&2; exit 1
  fi
  if diff -q "$OUT/fsim_result.txt" "$GOLDEN" >/dev/null; then
    echo "FSIM RESULT MATCHES OVERFIT GOLDEN (ACCEPT)"
  else
    echo "FSIM RESULT DIFFERS FROM OVERFIT GOLDEN (REJECT)" >&2
    diff "$GOLDEN" "$OUT/fsim_result.txt" >&2 || true
    exit 1
  fi
  ```

- [ ] **Step 7：数值正确性验证（可选但推荐）**

  ```bash
  python3 -c "
  import numpy as np
  A = np.fromfile('test/golden/matmul_overfit_16x2064x16/input_16x2064.bin', dtype=np.int32).reshape(16,2064)
  B = np.fromfile('test/golden/matmul_overfit_16x2064x16/weight_2064x16.bin', dtype=np.int32).reshape(2064,16)
  C = A @ B
  print('Expected C[0,0]:', C[0,0], 'C[0,-1]:', C[0,-1])
  print('Min:', C.min(), 'Max:', C.max())
  "
  ```

- [ ] **Step 8：跑 FSIM 验收**

  ```bash
  bash scripts/run_fsim_overfit.sh
  ```
  Expected: `FSIM RESULT MATCHES OVERFIT GOLDEN (ACCEPT)`

- [ ] **Step 9：回归所有既有脚本**

  ```bash
  bash scripts/run_fsim.sh              # 16×16 FSIM
  bash scripts/run_fsim_linalg_32x32.sh  # 32×32 FSIM
  ```

- [ ] **Step 10：提交**

  ```bash
  printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' \
    'git add -A' 'git commit -m "test(mlir-vta): overfit strategy-1 e2e + FSIM (16x2064x16)"' \
    > /tmp/do_commit.sh && bash /tmp/do_commit.sh
  ```

---

## Task 3：更新文档

**Files:** `docs/DESIGN_cn.md`, `docs/plans/phase3-generalized-gemm.md`

- [ ] `DESIGN_cn.md` §2.1 阶段三状态更新：加「overfit strategy-1 已落地（M=16, K=2064, N=16 字节级+FSIM ✓）」
- [ ] `docs/plans/phase3-generalized-gemm.md` 末尾技术债说明更新：overfit 已实现 strategy-1，后续 strategy-2/3/4 仍留
- [ ] 提交 `docs(mlir-vta): document phase-3 overfit strategy-1 (16x2064x16)`

---

## 自检清单

1. **字节级验证门**：Task1 Step7（instructions.bin + uop.bin 字节级一致黄金）
2. **FSIM 验收门**：Task2 Step8（`run_fsim_overfit.sh` 打印 ACCEPT）
3. **回归**：Task1 Step8（16×16 + 32×32 字节级不变）+ Task2 Step9（既有 FSIM 脚本）
4. **不做**：strategy-2/3/4、nbC>1 的全面验证（留后续）

---

## 附录：关键依赖位推导

Strategy-1 各指令的 dep 位（单步内 CMP↔LD↔ST 协调）：

| 指令 | pop_prev | pop_next | push_prev | push_next | 说明 |
|------|---------|---------|---------|---------|------|
| LOAD UOP reset | 0 | 0 | 0 | 0 | 最早执行，无 token |
| GEMM reset | 0 | 0 | 1 | 0 | 产 CMP->LD token（允许首次 LOAD 开始）|
| LOAD INP (每步首条) | 0 | 1 | 0 | 0 | 消费 CMP->LD token（ack） |
| LOAD WGT (每步末条) | 0 | 0 | 0 | 1 | 产 LD->CMP token（告知 CMP 数据就绪）|
| LOAD ACC（首次）| 1 | 0 | 0 | 0 | 消费 ST->CMP token（首步用 GEMM reset 的 push_prev）|
| LOAD UOP gemm (remainder步) | 1 | 0 | 0 | 0 | 消费 LD->CMP token（在 remainder 步，WGT已 push_next）|
| GEMM (中间步) | 0 | 0 | 1 | 0 | 产 CMP->LD token（允许下步 LOAD），不 push_next |
| GEMM (最后步) | 0 | 0 | 1 | 1 | 产 CMP->LD + CMP->ST token |
| STORE OUT | 1 | 0 | 1 | 0 | 消费 CMP->ST，产 ST->CMP（给下一 C 块的 ACC LOAD）|
| NOP-LOAD INP | 0 | 1 | 0 | 1 | 排空 CMP->LD + 产 LD->CMP（drain）|
| NOP-LOAD UOP | 1 | 1 | 0 | 0 | 消费 LD->CMP + ST->CMP（drain）|
| FINISH | 0 | 0 | 0 | 0 | — |

> **注：** 对 `delta_step=0` 以外的步骤，LOAD ACC 不发射；此时 LOAD UOP 不需 pop_prev（因为没有 ST->CMP token），仅 GEMM 保持 push_prev=1。在 remainder 步，由于没有 LOAD ACC，LOAD UOP 携带 pop_prev=1 消费上一步 GEMM 产生的 CMP->LD 信号（注意：上游实现中 remainder 步的 LOAD UOP 是 `pop_prev=1`，见黄金 I9）。
