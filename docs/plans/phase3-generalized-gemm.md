# MLIR-VTA 第三阶段实现计划 · 通用维度 GEMM（先打通 32×32×32）

> **For agentic workers:** 按 task 顺序实施；每个 task 末尾有**对齐上游黄金的验证门**。Steps 用 `- [ ]` 跟踪。

**Goal:** 把 `vta.gemm` 从 16×16 单块扩展到 16 倍数维度（先 32×32×32 / 2×2 块），端到端
`linalg.matmul`(32×32 tensor) → bufferize → `vta.gemm`(32×32 memref) → 通用化 `lower-vta-gemm` → `vta-translate`（多块数据/CSV）→ FSIM，
使结果矩阵与上游黄金 `test/golden/matmul_32x32/fsim_result_32x32.txt` 逐元素一致。

**设计依据：** spec [`../specs/phase3-generalized-gemm-design.md`](../specs/phase3-generalized-gemm-design.md)；黄金解码 [`spike-generalized-gemm-notes.md`](spike-generalized-gemm-notes.md)。

**关键约束（不变）：** lowering 内部复刻上游块调度（CASE 1 单 step），**不**用 MLIR `linalg-tile`；沿用固定缓冲逻辑基址（INP=64/WGT=8/ACC=192/OUT=256/UOP=5120）按块算偏移；4 计数器信号量推导依赖位。

## 环境与 Git（沿用前阶段约定）

- Git 根 `/mnt/c/MLIR-VTA/mlir-vta`，分支 `main`。提交走临时脚本规避 `--trailer` 注入：
  ```bash
  printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' \
    'git add -A' 'git commit -m "<msg>"' > /tmp/do_commit.sh && bash /tmp/do_commit.sh
  ```
- 增量构建：`cmake --build ~/mlir-vta-build -j"$(nproc)" --target vta-opt vta-translate`
- 改 CMake 后先重配（参数见 phase2 计划）。

## 黄金（Task 0 已完成）

`test/golden/matmul_32x32/` 已含：`instructions.bin`(14×16B)、`uop.bin`(9×4B)、`input.bin`/`weight.bin`/`accumulator.bin`(各 4096B)、
`memory_addresses.csv`、`metadata.csv`、原始 `input_32x32.bin`/`weight_32x32.bin`、`fsim_result_32x32.txt`、`matmul_32x32.json`。
解码后的 14 条指令 + 9 条 uop 见 spike notes。

---

### Task 1：数据发射器多块化（可独立先验，先攻最易验证的一环）

**目的：** 让 `vta-translate --emit-data` 接受 N×N 维度，按「块主序 + WGT 每块转置 + nbC 块零 ACC/OUT」产出，对黄金 `cmp` 一致。这一环不依赖 lowering，可独立验证，先做。

**Files:**
- Modify: `lib/Target/VTADataEmitter.cpp`（按维度参数化）
- Modify: `include/mlir-vta/Target/VTADataEmitter.h`（`emitData` 增加 `rows,cols,k` 块维或元素维参数）
- Modify: `tools/vta-translate/vta-translate.cpp`（新增 `--rows --cols --k` 元素维 CLI，默认 16）

- [ ] **Step 1：扩展 `emitData` 签名**接收元素维 `mRows,kCols,nCols`（均 16 倍数）。读 `inputPath`=M×K、`weightPath`=K×N 原始行主序全矩阵（按文件应有字节数读取，不再写死 1024）。
- [ ] **Step 2：块主序写出**：
  - `input.bin`：对 `bi in [0,M/16) × bk in [0,K/16)` 行主序遍历，每块 `A[16bi:+16, 16bk:+16]` 行主序拼接，**不转置**。
  - `weight.bin`：对 `bk in [0,K/16) × bj in [0,N/16)` 行主序遍历，每块 `B[16bk:+16,16bj:+16]` **转置**后拼接。
  - `accumulator.bin`/`out_init.bin`：`(M/16)*(N/16)*256` 个 int32 零。
- [ ] **Step 3：CSV 参数化**：`metadata.csv` 的 `A` 行写 `M,K`、`X`/`C` 行写 `M,N`（照黄金 `metadata.csv`：`A,32,32` / `X,32,32,False` / `C,32,32,True`）。`memory_addresses.csv` 暂保持 6 行固定基址（与黄金一致）。`layers_name.csv` 复用。
- [ ] **Step 4：vta-translate CLI**：加 `--rows <M> --cols <N> --k <K>`（默认 16），传入 `emitData`。保持 16×16 旧调用（无新 flag）行为不变。
- [ ] **Step 5：构建** `--target vta-translate`。
- [ ] **Step 6：验证门（对齐黄金，字节级）**
  ```bash
  rm -rf /tmp/d32 && mkdir -p /tmp/d32
  ~/mlir-vta-build/bin/vta-translate test/Target/gemm16x16.mlir -o /tmp/d32 \
    --emit-data --input test/golden/matmul_32x32/input_32x32.bin \
    --weight test/golden/matmul_32x32/weight_32x32.bin --rows 32 --cols 32 --k 32
  cmp /tmp/d32/input.bin       test/golden/matmul_32x32/input.bin       && echo "INP DATA OK"
  cmp /tmp/d32/weight.bin      test/golden/matmul_32x32/weight.bin      && echo "WGT DATA OK"
  cmp /tmp/d32/accumulator.bin test/golden/matmul_32x32/accumulator.bin && echo "ACC DATA OK"
  cmp /tmp/d32/metadata.csv    test/golden/matmul_32x32/metadata.csv    && echo "META OK"
  ```
  Expected: 四个 OK。（指令 bin 用的是 gemm16x16.mlir，本 task 不关心；只验数据。）
- [ ] **Step 7：回归** 16×16 旧数据仍一致：`run_fsim.sh` 或 `cmp` 原 `test/golden/input_16x16` 路径产物。
- [ ] **Step 8：提交** `feat(mlir-vta): data emitter handles multi-block (block-major + per-block WGT transpose)`。

---

### Task 2：`vta.gemm` verifier 放宽 + `convert-linalg-to-vta` 接受 N×N

**Files:**
- Modify: `include/mlir-vta/Dialect/VTA/VTAOps.td`
- Modify: `lib/Transforms/ConvertLinalgToVTA.cpp`
- Modify: `test/Dialect/vta_gemm_roundtrip.mlir`（加 32×32 用例）

- [ ] **Step 1：verifier 放宽**：三 operand 均 2D memref、每维 16 倍数、`lhs:MxK / rhs:KxN / acc:MxN` 维度匹配。非此报错。（删去「仅 16×16」硬断言。）
- [ ] **Step 2：`convert-linalg-to-vta`** 去掉对 16×16 的尺寸限制，接受任意已 bufferize 的 `linalg.matmul`（仍要求 buffer 语义）。
- [ ] **Step 3：构建** `--target vta-opt`。
- [ ] **Step 4：验证**
  ```bash
  printf 'func @m(%%a: memref<32x32xi32>, %%b: memref<32x32xi32>, %%c: memref<32x32xi32>) {\n  vta.gemm ins(%%a, %%b : memref<32x32xi32>, memref<32x32xi32>) outs(%%c : memref<32x32xi32>)\n  return\n}\n' > /tmp/g32.mlir
  ~/mlir-vta-build/bin/vta-opt /tmp/g32.mlir | grep -q "vta.gemm" && echo "32x32 ROUNDTRIP OK"
  # 非 16 倍数仍应被拒
  printf 'func @m(%%a: memref<17x16xi32>, %%b: memref<16x16xi32>, %%c: memref<17x16xi32>) {\n  vta.gemm ins(%%a, %%b : memref<17x16xi32>, memref<16x16xi32>) outs(%%c : memref<17x16xi32>)\n  return\n}\n' > /tmp/gbad.mlir
  ~/mlir-vta-build/bin/vta-opt /tmp/gbad.mlir 2>&1 | grep -q "16" && echo "REJECT NON-16x OK"
  ```
- [ ] **Step 5：提交** `feat(mlir-vta): relax vta.gemm verifier to 16-multiple dims; convert accepts NxN`。

---

### Task 3：通用化 `lower-vta-gemm`（块调度 + uop + 合并 LOAD + 单 GEMM + 逐块 STORE + 依赖位）

**这是核心 task。** 把 `LowerVTAGemm.cpp` 由「硬编码 11 条」改为按 operand 形状计算的通用发射。建议拆出辅助：`GemmScheduler`（产块调度与 uop）、`SemaphoreState`（4 计数器依赖位）。

**Files:**
- Modify: `lib/Transforms/LowerVTAGemm.cpp`

- [ ] **Step 1：读形状求块维**：`Mb=M/16, Kb=K/16, Nb=N/16`；`nbA=Mb*Kb, nbB=Kb*Nb, nbC=Mb*Nb`。本阶段断言无 overfit（块数远小于容量），否则 emitError（overfit 留后续）。
- [ ] **Step 2：块调度（复刻 `get_gemm_operations`）**：行主序，产 GeMM 列表 `[(c_idx,a_idx,b_idx)]`：
  ```
  for a in [0,nbA): i=a/Kb, k=a%Kb
    for b in [0,nbB): bk=b/Nb, j=b%Nb
      if bk==k: emit (i*Nb+j, a, b)
  ```
  （遍历顺序须与上游一致：外层 a 升序、内层 b 升序 → 对 32×32 得 8 条且与 uop.bin 顺序一致。）
- [ ] **Step 3：UOP 表**：第 0 条 reset 占位 `(0,0,0)`；随后每条 GeMM 一条：`dst=c_idx*16, src=a_idx*16, wgt=b_idx`。`G=len(gemm)`。用单个 `vtaisa.uop_table`（dst/src/wgt 三个等长数组，长度 `1+G`）。
- [ ] **Step 4：发射 vtaisa 指令骨架**（dram_base 用固定逻辑基址，块偏移见下）：
  | 指令 | 取值 |
  |------|------|
  | LOAD UOP reset | dram=5120, y=1, x=1, stride=1 |
  | GEMM reset | reset=1, uop[0,1), lp_out=1, lp_in=16, dst_factor_out=16, dst_factor_in=1 |
  | LOAD INP | dram=64, y=nbA, x=16, stride=16 |
  | LOAD WGT | dram=8, y=nbB, x=1, stride=1 |
  | LOAD ACC | dram=192, y=nbC, x=16, stride=16 |
  | LOAD UOP gemm | dram=5121, y=1, x=G, stride=G |
  | GEMM | reset=0, uop[0,G), lp_out=1, lp_in=16, dst_factor_in=1, src_factor_in=1 |
  | STORE OUT ×nbC | blk in [0,nbC): sram=blk*16, dram=256+blk*16, y=1, x=16, stride=16 |
  | NOP-LOAD (INP) | y=0,x=0,dram=0 |
  | NOP-COMPUTE (UOP) | y=0,x=0,dram=0 |
  | FINISH | |
- [ ] **Step 5：依赖位（4 计数器 `SemaphoreState`）**：按 spec §6 规则设置并更新计数器。**先以单 step 序列对齐 spike §解码表逐位**，再确认抽象逻辑得同样结果。单 step 目标位（务必逐位匹配）：
  ```
  GEMM reset: push_prev=1
  LOAD INP : pop_next=1
  LOAD WGT : push_next=1
  LOAD ACC : pop_prev=1
  GEMM     : push_prev=1, push_next=1
  STORE[0] : pop_prev=1 ; STORE[last]: push_prev=1
  NOP-LOAD : pop_next=1, push_next=1
  NOP-COMP : pop_prev=1, pop_next=1
  ```
- [ ] **Step 6：构建** `--target vta-opt vta-translate`。
- [ ] **Step 7：验证门（字节级，对齐黄金）**
  ```bash
  ~/mlir-vta-build/bin/vta-opt -lower-vta-gemm /tmp/g32.mlir -o /tmp/low32.mlir
  rm -rf /tmp/o32 && mkdir -p /tmp/o32
  ~/mlir-vta-build/bin/vta-translate /tmp/low32.mlir -o /tmp/o32
  cmp /tmp/o32/instructions.bin test/golden/matmul_32x32/instructions.bin && echo "INSN OK (14 insn)"
  cmp /tmp/o32/uop.bin          test/golden/matmul_32x32/uop.bin          && echo "UOP OK (9 uop)"
  ```
  Expected: `INSN OK` + `UOP OK`（这是 Task 3 的硬验收）。
- [ ] **Step 8：回归** 16×16：`-lower-vta-gemm test/Target/lower_gemm.mlir | vta-translate` → `cmp test/golden/instructions.bin`（通用代码在 Mb=Kb=Nb=1 时须退化为第一阶段 11 条）。
- [ ] **Step 9：提交** `feat(mlir-vta): generalized lower-vta-gemm (multi-block schedule + semaphore derivation)`。

---

### Task 4：端到端入口 + FSIM 验收门

**Files:**
- Create: `test/Target/matmul_tensor_32x32.mlir`
- Create: `scripts/run_fsim_linalg_32x32.sh`

- [ ] **Step 1：入口 IR**（仿 phase2 的 16×16，把维度换 32×32；用 std 风格 `constant`）：
  ```mlir
  func @main(%A: tensor<32x32xi32>, %B: tensor<32x32xi32>) -> tensor<32x32xi32> {
    %c0 = constant 0 : i32
    %init = linalg.init_tensor [32, 32] : tensor<32x32xi32>
    %acc = linalg.fill(%c0, %init) : i32, tensor<32x32xi32> -> tensor<32x32xi32>
    %out = linalg.matmul ins(%A, %B : tensor<32x32xi32>, tensor<32x32xi32>)
                         outs(%acc : tensor<32x32xi32>) -> tensor<32x32xi32>
    return %out : tensor<32x32xi32>
  }
  ```
- [ ] **Step 2：管道**（**无 linalg-tile**）：
  ```bash
  ~/mlir-vta-build/bin/vta-opt test/Target/matmul_tensor_32x32.mlir \
    --linalg-bufferize --tensor-bufferize --tensor-constant-bufferize \
    --func-bufferize --finalizing-bufferize \
    --convert-linalg-to-vta --canonicalize --lower-vta-gemm \
    -o /tmp/low_e2e32.mlir
  grep -q "vtaisa.finish" /tmp/low_e2e32.mlir && echo "HAS vtaisa"
  grep -q "linalg.matmul\|vta.gemm" /tmp/low_e2e32.mlir && echo "RESIDUAL (BAD)" || echo "no residual"
  ```
  > 若 bufferize 后是 `memref<32x32>` 的 `linalg.matmul`，convert 直接成 `vta.gemm`(32×32)；lower 走 Task 3 通用路径。
- [ ] **Step 3：`run_fsim_linalg_32x32.sh`**（仿 `run_fsim_linalg.sh`，数据用 `--rows 32 --cols 32 --k 32` + 原始 32×32 bin；黄金对 `fsim_result_32x32.txt` 自校验）。
- [ ] **Step 4：验收门**
  ```bash
  bash scripts/run_fsim_linalg_32x32.sh
  ```
  Expected: 打印 `FSIM RESULT MATCHES GOLDEN 32x32 (ACCEPT)`（结果矩阵 4 块 == 黄金）。
- [ ] **Step 5：提交** `test(mlir-vta): 32x32 generalized GEMM end-to-end + FSIM acceptance`。

---

### Task 5：更新设计文档

**Files:** Modify `docs/DESIGN_cn.md`、`README.md`、`docs/README.md`

- [ ] §2.1 阶段三状态由「规划中」→「进行中（通用 GEMM·32×32 多块已落地）」。
- [ ] §7 增「通用化 lowering：块调度 + uop + 合并 LOAD + 逐块 STORE + 4 计数器依赖位」。
- [ ] §10 增 32×32 端到端验证行；§11 技术债更新（overfit/allocation pass/ALU 仍未做）。
- [ ] README 当前状态表加阶段三行；docs/README 索引加 spec/plan/spike。
- [ ] 提交 `docs(mlir-vta): document phase-3 generalized GEMM (32x32)`。

---

## 自检清单

1. **Spec 覆盖**：§4 verifier→T2；§5 块调度/uop/骨架→T3；§6 依赖位→T3 Step5；§7 数据→T1；§10 验收→T4；§12 文档→T5。
2. **字节级验证门**：T1 数据 bin、T3 instructions/uop bin、T4 FSIM 结果——三道均对齐 `test/golden/matmul_32x32/`。
3. **回归**：T1/T3 均要求 16×16 旧用例仍字节一致（通用代码在 1×1 块退化为第一阶段）。
4. **不做**：overfit、独立 allocation pass、ALU——留后续增量。

## 已知风险与缓解

| 风险 | 缓解 |
|------|------|
| bufferize 32×32 `linalg.matmul` 形态与 16×16 不同（残留 copy/alloc） | T4 Step2 先手验；emitter 已忽略非 vtaisa op；必要时调 pass 序列 |
| 通用 lowering 在 1×1 块未退化为第一阶段 11 条 | T3 Step8 强制回归 `cmp` 16×16 golden |
| 依赖位抽象逻辑与单 step 黄金不一致 | T3 Step5 先逐位对齐黄金，再抽象；以 `cmp instructions.bin` 为准 |
| UOP `uop_bgn/uop_end` 语义（reset uop 占位 vs gemm uop 区间） | 以黄金字节为准：reset GEMM uop[0,1)、gemm GEMM uop[0,G)，LOAD UOP 分两条（reset@5120 x=1；gemm@5121 x=G） |
