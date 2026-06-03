# MLIR-VTA 阶段三增量：ALU Lowering（`vta.alu` + `lower-vta-alu`）

> **实施状态：✅ 已落地（2026-06）。** Task 0–4 全部完成：黄金生成、`vta.alu` op 定义、`computeAluLayout` 地址计算、`lower-vta-alu` pass 实现、字节级验收通过（ADD_IMM 16×16：24 insns + 17 uops 对齐上游黄金）。提交 `e331595`。

**Goal:** 实现高层 `vta.alu` op 及对应的 `--lower-vta-alu` pass，把 ALU 运算（ADD_IMM / MAX / SHR 等）展开为 `vtaisa.alu_insn` + UOP 表 + LOAD/STORE 指令序列，字节级对齐上游 standalone-vta `alu_16x16` 黄金。

**架构：**
- 新增高层 `vta.alu` op（`VTAOps.td`）：acc memref [M,16] + alu_opcode/use_imm/imm 属性。
- 新增 `computeAluLayout`（`VTAGemmLayout.h`）：DRAM 页对齐顺序 ACC/OUT/UOP/INSN（无 INP/WGT），复刻上游 `dram_allocation`。
- 新增 `LowerVTAAlu.cpp`：按行发射 UOP 表 + `GEMM reset` + `LOAD ACC` + `LOAD UOP(alu)` + 主 `ALU insn` + M 条 `STORE` + 2 条 NOP + `FINISH`。

**设计依据：** 上游 `standalone-vta/examples/vta_ir/alu_16x16.json`（ADD_IMM）、`standalone-vta/src/compiler/vta_compiler/matrix_partitioning/alu_strategies.py`、黄金 `test/golden/alu_add_imm_16x16/`（24 insns, 17 uops）。

---

## 关键数学——ALU 指令序列规律

### 指令序列（共 8+M 条）

| 顺序 | 指令 | 关键参数 | 依赖位 |
|------|------|---------|--------|
| 0 | `LOAD UOP(reset)` | dram=uopBase, y=1, x=1, stride=1 | — |
| 1 | `GEMM reset` | uop[0,1), lp_out=1, lp_in=16, dst_factor=[16,1] | push_prev=1 |
| 2 | `LOAD ACC` | dram=accBase, y=ceil(M/16), x=16, stride=16 | — |
| 3 | `LOAD UOP(alu)` | dram=uopBase+1, y=1, x=M, stride=M | — |
| 4 | `ALU main` | uop[0,M), lp_out=1, lp_in=1, use_imm, imm, alu_opcode | push_next=1 |
| 5..4+M | `STORE OUT` × M | sram=i, dram=outBase+i, y=1, x=1, stride=1 | STORE[0]:pop_prev=1; STORE[M-1]:push_prev=1 |
| 5+M | `NOP-LOAD(INP)` | y=0, x=0, dram=0 | pop_next=1, push_next=1 |
| 6+M | `NOP-LOAD(UOP)` | y=0, x=0, dram=0 | pop_prev=1, pop_next=1 |
| 7+M | `FINISH` | | |

> ADD_IMM 16×16：M=16，共 24 条指令。

### UOP 表（1 reset + M 条）

```
uop[0]   = (0, 0, 0)        ← reset 占位
uop[1+i] = (i, 0, 0)  i∈[0,M)  ← dst=row i, src=0, wgt=0
```

总共 1+M 条，17 条（M=16 时）。

`LOAD UOP(reset)` 加载 uop[0] 到 SRAM[0]；`LOAD UOP(alu)` 加载 uop[1..M] 到 SRAM[0..M-1]（覆盖 reset），`ALU main` 引用 SRAM uop[0..M-1]（`uop_bgn=0, uop_end=M`）。

### DRAM 地址布局（`computeAluLayout`，4 KiB 页对齐，无 INP/WGT）

对 M=16 行（nbBlocks=1）：

| 缓冲 | 物理基址 | 逻辑基址 | 大小 |
|------|---------|---------|------|
| ACC  | 0x1000  | 0x40    | 1×1024 B |
| OUT  | 0x2000  | 0x80    | 1×1024 B |
| UOP  | 0x3000  | 0xc00   | (1+M)×4 B |
| INSN | 0x4000  | 0x400   | (8+M)×16 B |

逻辑基址 = 物理基址 / divisor：ACC/OUT divisor=64；UOP divisor=4；INSN divisor=16。

---

## Task 0：生成并保存 ALU 黄金

**Files:**
- Create: `test/golden/alu_add_imm_16x16/instructions.bin`（24×16B）
- Create: `test/golden/alu_add_imm_16x16/uop.bin`（17×4B）
- Create: `test/golden/alu_add_imm_16x16/memory_addresses.csv`

- [x] **Step 1：准备输入数据**
  ```python
  # standalone-vta 目录
  import numpy as np
  np.random.seed(42)
  data = np.random.randint(-128, 127, (16, 16), dtype=np.int32)
  data.tofile('compiler_output/accumulator_16x16.bin')
  ```

- [x] **Step 2：运行上游编译器**
  ```bash
  cd /mnt/c/MLIR-VTA/standalone-vta/examples
  python3 ../src/compiler/vta_compiler/main_vta_compiler.py True True False \
    ../config/vta_config.json vta_ir/alu_16x16.json
  ```
  Expected: `nb_steps=1, nb_uop=17, nb_insn=24`

- [x] **Step 3：复制黄金**
  ```bash
  GD=/mnt/c/MLIR-VTA/mlir-vta/test/golden/alu_add_imm_16x16
  mkdir -p $GD
  cp compiler_output/instructions.bin $GD/
  cp compiler_output/uop.bin $GD/
  cp compiler_output/memory_addresses.csv $GD/
  ```

---

## Task 1：定义 `vta.alu` 高层 op

**Files:**
- Modify: `include/mlir-vta/Dialect/VTA/VTAOps.td`

- [x] **Step 1：在 VTAOps.td 末尾（`#endif` 前）加入 `VTA_AluOp`**
  ```tablegen
  def VTA_AluOp : VTA_Op<"alu"> {
    let summary = "element-wise ALU on acc memref: C = op(acc, imm) or op(acc, src)";
    let arguments = (ins AnyMemRef:$acc,
                         I64Attr:$alu_opcode,
                         DefaultValuedAttr<BoolAttr, "false">:$use_imm,
                         DefaultValuedAttr<I64Attr, "0">:$imm,
                         OptionalAttr<StrAttr>:$name);
    let results = (outs);
    let assemblyFormat =
      "`ins` `(` $acc `:` type($acc) `)` attr-dict";
    let verifier = [{
      auto accT = acc().getType().dyn_cast<::mlir::MemRefType>();
      if (!accT || accT.getRank() != 2)
        return emitOpError("vta.alu requires a 2-D memref operand");
      if (accT.getShape()[0] <= 0 || accT.getShape()[1] != 16)
        return emitOpError("vta.alu acc memref must have shape [M, 16]");
      int64_t op = alu_opcode();
      if (op < 0 || op > 4)
        return emitOpError("vta.alu alu_opcode must be in [0,4], got ") << op;
      return ::mlir::success();
    }];
  }
  ```

- [x] **Step 2：构建验证**
  ```bash
  cmake --build ~/mlir-vta-build -j"$(nproc)" --target vta-opt 2>&1 | tail -3
  ```

- [x] **Step 3：roundtrip 测试**
  ```bash
  printf 'func @t(%%a: memref<16x16xi32>) {\n  vta.alu ins(%%a : memref<16x16xi32>) {alu_opcode = 2 : i64, use_imm = true, imm = 3 : i64}\n  return\n}\n' > /tmp/alu_rt.mlir
  ~/mlir-vta-build/bin/vta-opt /tmp/alu_rt.mlir | grep -q "vta.alu" && echo "ROUNDTRIP OK"
  ```

---

## Task 2：实现 `computeAluLayout`

**Files:**
- Modify: `include/mlir-vta/VTAGemmLayout.h`（在 `computeGemmLayout` 后追加）

- [x] **Step 1：添加 `AluLayout` 结构体和 `computeAluLayoutAt`**

  关键：
  - 缓冲顺序：ACC → OUT → UOP → INSN（无 INP/WGT）
  - `nbBlocks = ceil(nbVec / 16)`，每块 1024 B
  - UOP size = (1 + G) × 4 B（G = nbVec = M）
  - numInsns = 8 + nbVec（LOAD UOP reset + GEMM reset + LOAD ACC + LOAD UOP alu + ALU main + M STORE + 2 NOP + FINISH）

- [x] **Step 2：验证地址（M=16 时应与黄金 CSV 一致）**

  ```python
  # ACC: nextPage(0)=0x1000, logical=0x1000/64=0x40  ✓
  # OUT: nextPage(0x13FF)=0x2000, logical=0x80        ✓
  # UOP: nextPage(0x23FF)=0x3000, logical=0x3000/4=0xc00 ✓
  ```

---

## Task 3：实现 `lower-vta-alu` pass

**Files:**
- Create: `lib/Transforms/LowerVTAAlu.cpp`
- Modify: `lib/Transforms/CMakeLists.txt`（加入 `LowerVTAAlu.cpp`）
- Modify: `lib/Transforms/LowerVTAGemm.cpp`（`registerVTAPasses` 注册新 pass）
- Modify: `include/mlir-vta/Dialect/VTA/VTAPasses.h`（声明 `createLowerVTAAluPass()`）

- [x] **Step 1：实现 pass（关键代码片段）**

  ```cpp
  // UOP table
  uopDst.push_back(0); uopSrc.push_back(0); uopWgt.push_back(0); // reset
  for (int64_t i = 0; i < G; ++i) { uopDst.push_back(i); uopSrc.push_back(0); uopWgt.push_back(0); }
  b.create<vtaisa::UopTableOp>(loc, uopDst, uopSrc, uopWgt);

  // LOAD UOP reset
  b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, false,false,false,false,
      0, L.uopLogical, 1, 1, 1);
  // GEMM reset (push_prev=1)
  b.create<vtaisa::GemmInsnOp>(loc, false,false,true,false, true, 0,1, 1,16, 16,1, 0,0);
  // LOAD ACC
  b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::ACC, false,false,false,false,
      0, L.accLogical, nbBlocks, 16, 16);
  // LOAD UOP alu
  b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, false,false,false,false,
      0, L.uopLogical+1, 1, G, G);
  // ALU main (push_next=1)
  b.create<vtaisa::AluInsnOp>(loc, false,false,false,true, false, 0,G, 1,1, 0,0,0,0,
      op.alu_opcode(), op.use_imm(), op.imm());
  // M × STORE
  for (int64_t i = 0; i < nbVec; ++i)
    b.create<vtaisa::StoreOp>(loc, vtaisa::BufferId::OUT,
        /*pop_prev=*/(i==0), false, /*push_prev=*/(i==nbVec-1), false,
        i, L.outLogical+i, 1, 1, 1);
  // NOP-LOAD(INP): pop_next=1, push_next=1
  b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP, false,true,false,true, 0,0,0,0,0);
  // NOP-LOAD(UOP): pop_prev=1, pop_next=1
  b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, true,true,false,false, 0,0,0,0,0);
  b.create<vtaisa::FinishOp>(loc);
  op.erase();
  ```

- [x] **Step 2：构建**
  ```bash
  cmake --build ~/mlir-vta-build -j"$(nproc)" --target vta-opt vta-translate 2>&1 | tail -5
  ```

---

## Task 4：字节级验收 + 回归

- [x] **Step 1：验证门（字节级对齐黄金）**
  ```bash
  GD=/mnt/c/MLIR-VTA/mlir-vta/test/golden/alu_add_imm_16x16
  printf 'func @main(%%acc: memref<16x16xi32>) {\n  vta.alu ins(%%acc : memref<16x16xi32>) {alu_opcode = 2 : i64, use_imm = true, imm = 3 : i64}\n  return\n}\n' > /tmp/alu16x16.mlir
  ~/mlir-vta-build/bin/vta-opt --lower-vta-alu /tmp/alu16x16.mlir -o /tmp/alu16_low.mlir
  rm -rf /tmp/alu_out && mkdir -p /tmp/alu_out
  ~/mlir-vta-build/bin/vta-translate /tmp/alu16_low.mlir -o /tmp/alu_out
  cmp /tmp/alu_out/instructions.bin $GD/instructions.bin && echo "ALU INSN OK (24 insns)"
  cmp /tmp/alu_out/uop.bin          $GD/uop.bin          && echo "ALU UOP OK (17 uops)"
  ```
  Expected: 两个 OK。

- [x] **Step 2：回归（16×16 GEMM 不受影响）**
  ```bash
  rm -rf /tmp/reg16 && mkdir -p /tmp/reg16
  ~/mlir-vta-build/bin/vta-opt -lower-vta-gemm test/Target/lower_gemm.mlir -o /tmp/reg16/low.mlir
  ~/mlir-vta-build/bin/vta-translate /tmp/reg16/low.mlir -o /tmp/reg16
  cmp /tmp/reg16/instructions.bin test/golden/instructions.bin && echo "16x16 GEMM PASS"
  ```

- [x] **Step 3：提交**
  ```bash
  printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' \
    'git add include/mlir-vta/Dialect/VTA/VTAOps.td include/mlir-vta/Dialect/VTA/VTAPasses.h' \
    'git add include/mlir-vta/VTAGemmLayout.h lib/Transforms/CMakeLists.txt' \
    'git add lib/Transforms/LowerVTAGemm.cpp lib/Transforms/LowerVTAAlu.cpp' \
    'git add test/golden/alu_add_imm_16x16/' \
    'git commit -m "feat(mlir-vta): ALU lowering (vta.alu + lower-vta-alu, ADD_IMM 16x16 byte-verified)"' \
    > /tmp/do_commit.sh && bash /tmp/do_commit.sh
  ```

---

## 自检清单

1. **字节级验证门**：Task 4 Step 1 的两个 `cmp` 均通过（24 insns + 17 uops）
2. **回归**：16×16 GEMM `instructions.bin` 字节级不变
3. **roundtrip**：`vta.alu` 属性打印/解析正确
4. **alu_opcode 范围**：verifier 拒绝 opcode ∉ [0,4]

---

## 附录：上游 ALU IR 格式

```json
{
  "NAME": "",
  "MATRICES": { "X": [16, 16, "...accumulator_16x16.bin"], "C": [16, 16, "output"] },
  "LOAD": { "ACC": ["X"] },
  "ALU":  { "C": [["ADD_IMM", [[0, 1], 3, 16]]] },
  "STORE": { "C": ["C"] }
}
```

`ADD_IMM [[row_start, row_end], shift, row_count]`：对第 0 行（row_start=0, row_end=1）的 row_count=16 个元素做 `x >> shift` 后 bias-shift（opcode=2 ADD，use_imm=true，imm=shift=3）。整个 16×16 矩阵按行逐条展开为 16 条 ALU UOP，与上游 `alu_strategy` 的 `sorted_alu_ops` 输出一致。

## 后续增量

- MAX pool（opcode=1）、SHR（opcode=3）：与 ADD_IMM 结构相同，仅 `alu_opcode` 和 `use_imm/imm` 不同，同一 pass 可直接支持，需补充黄金验收。
- 真·层间串联（增量 C）：依赖 `fsim_nn` + `dependency.csv`，推迟。
