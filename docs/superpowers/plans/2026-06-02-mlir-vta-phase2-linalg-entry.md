# MLIR-VTA 第二阶段实现计划 · `linalg.matmul` 入口（16×16 单块）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 16×16 单块用例上立起一条真实 MLIR 中端链路 `linalg.matmul`(tensor) → `--linalg-tile` → bufferization → `vta.gemm`(memref operand) → 复用第一阶段单块 lowering → `vta-translate` → FSIM，使 FSIM 结果矩阵与第一阶段（numpy 参考）逐元素一致。

**Architecture:** 扩展 `vta.gemm` 承载 memref operand；新增 `--convert-linalg-to-vta` pass 把 bufferize 后的 `linalg.matmul` 重写为 `vta.gemm`；让 `vta-opt` 注册全部上游 dialect/pass（取得 `-linalg-tile`/`*-bufferize`/`-canonicalize`）；让二进制发射器递归遍历以容忍 op 嵌在 `func`/`scf` 内。数据/CSV 发射器与单块 lowering 的黄金地址/依赖位完全复用、不改。

**Tech Stack:** C++17、MLIR/LLVM 13.0.0（`/usr/local/llvm`）、CMake 3.22 + Unix Makefiles、`clang++`、TableGen、上游 `mlir-opt`（spike 用）、standalone-vta FSIM（黄金参考与验收）。

---

## 仓库布局与 Git（沿用第一阶段约定）

- **Git 仓库根 = `/mnt/c/MLIR-VTA/mlir-vta/`**，分支 `main`。`standalone-vta/` 是独立上游，不纳入本仓库。
- 提交统一在仓库根执行：`cd /mnt/c/MLIR-VTA/mlir-vta && git add -A && git commit -m "<msg>"`。
- **已知环境坑：本机 git 2.25.1，外层包装会向 `git commit` 注入 `--trailer`（2.32+ 才支持）导致直接失败。** 规避：把 `git add`/`git commit` 写进一个临时脚本再 `bash` 执行，例如：
  ```bash
  printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' \
    'git add -A' 'git commit -m "<msg>"' > /tmp/do_commit.sh && bash /tmp/do_commit.sh
  ```
- build 目录在 `~/mlir-vta-build`（不占 `/mnt/c`）。增量构建命令：
  ```bash
  cmake --build ~/mlir-vta-build -j"$(nproc)" --target vta-opt vta-translate
  ```
- 若改了 `CMakeLists.txt`，先重新配置一次（保留第一阶段的配置参数）：
  ```bash
  cmake -S /mnt/c/MLIR-VTA/mlir-vta -B ~/mlir-vta-build -G "Unix Makefiles" \
    -DMLIR_DIR=/usr/local/llvm/lib/cmake/mlir -DLLVM_DIR=/usr/local/llvm/lib/cmake/llvm \
    -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_BUILD_TYPE=Release
  ```

## 验收标准（来自 spec）

- 端到端验收门：`matmul_tensor.mlir` 跑完整管道 → `vta-translate` → FSIM 结果矩阵 == 第一阶段同 fixture 的 FSIM 结果（即 `scripts/run_fsim.sh` 的输出矩阵）。
- 非阻断信号：新链路产出的 `instructions.bin`/`uop.bin` 对现有 golden 做 `cmp`（lowering 未变，理应仍一致）。
- 逐 pass IR 检查用 `vta-opt ... | grep -q`（环境无 FileCheck）。

## 文件结构（创建/修改一览）

| 文件 | 动作 | 职责 |
|------|------|------|
| `docs/superpowers/plans/_spike_bufferize_notes.md` | 创建 | 记录 spike 验证出的可用 pass 管道与样例 IR |
| `test/Target/_spike_bufferized.mlir` | 创建 | spike 产出的 bufferize 后 IR，作为 convert pass 测试输入 |
| `include/mlir-vta/Dialect/VTA/VTAOps.td` | 修改 | `vta.gemm` 改带 memref operand + verifier |
| `lib/Transforms/LowerVTAGemm.cpp` | 修改 | 守卫由读 `m,n,k` 属性改为读 memref 形状 |
| `lib/Transforms/ConvertLinalgToVTA.cpp` | 创建 | `linalg.matmul`(memref) → `vta.gemm` pass |
| `include/mlir-vta/Dialect/VTA/VTAPasses.h` | 修改 | 声明并注册新 pass |
| `lib/Transforms/CMakeLists.txt` | 修改 | 加新源文件 + 链接 `MLIRLinalg` |
| `tools/vta-opt/vta-opt.cpp` | 修改 | `registerAllDialects`/`registerAllPasses` |
| `tools/vta-opt/CMakeLists.txt` | 修改 | 链接全部 dialect/conversion 库 |
| `lib/Target/VTABinaryEmitter.cpp` | 修改 | 顶层遍历改递归 `module.walk` |
| `test/Target/gemm16x16_func.mlir` | 创建 | 把 13 条 op 包进 `func`，验证递归遍历 |
| `test/Target/matmul_tensor.mlir` | 创建 | tensor 级 linalg.matmul 入口 |
| `scripts/run_fsim_linalg.sh` | 创建 | 新入口的端到端 FSIM 跑法 |
| `docs/DESIGN_cn.md` | 修改 | §10.1 状态更新 + 数值约定边界 + spike 决策 |

---

### Task 0：管道 spike（确定可用的 tile + bufferization 命令；不改代码）

**目的：** 你选的方案 2（tile-on-tensors → bufferize）在 LLVM 13 上有成熟度风险。先用上游 `mlir-opt` 手跑，锁定可用的 pass 序列与 IR 形态；走不通则回退方案 1（先 bufferize 再在 memref 上 tile），并记录决策。本任务**不构建 vta-opt、不改 C++**。

**Files:**
- Create: `mlir-vta/test/Target/matmul_tensor.mlir`
- Create: `mlir-vta/test/Target/_spike_bufferized.mlir`（spike 输出）
- Create: `mlir-vta/docs/superpowers/plans/_spike_bufferize_notes.md`

- [ ] **Step 1：写入口 IR `matmul_tensor.mlir`**

```mlir
func @main(%A: tensor<16x16xi32>, %B: tensor<16x16xi32>) -> tensor<16x16xi32> {
  %c0 = arith.constant 0 : i32
  %init = linalg.init_tensor [16, 16] : tensor<16x16xi32>
  %acc = linalg.fill(%c0, %init) : i32, tensor<16x16xi32> -> tensor<16x16xi32>
  %out = linalg.matmul
           ins(%A, %B : tensor<16x16xi32>, tensor<16x16xi32>)
           outs(%acc : tensor<16x16xi32>) -> tensor<16x16xi32>
  return %out : tensor<16x16xi32>
}
```

- [ ] **Step 2：先确认入口能解析（语法对齐 LLVM 13）**

Run:
```bash
/usr/local/llvm/bin/mlir-opt mlir-vta/test/Target/matmul_tensor.mlir -o /dev/null && echo "PARSE OK"
```
Expected: 打印 `PARSE OK`。若 `linalg.fill`/`linalg.init_tensor` 语法报错，按报错调整为 LLVM 13 接受的写法（13 用 `linalg.init_tensor` 与 `linalg.fill(%v, %t)` 形态），重跑至 `PARSE OK`。

- [ ] **Step 3：方案 2 —— tensor 上 tile，再 bufferize**

Run（一次性管道；若某 pass 名/选项不被接受按 `--help` 修正）：
```bash
/usr/local/llvm/bin/mlir-opt mlir-vta/test/Target/matmul_tensor.mlir \
  --linalg-tile="linalg-tile-sizes=16,16,16" \
  --linalg-bufferize --tensor-bufferize --tensor-constant-bufferize \
  --func-bufferize --finalizing-bufferize \
  --canonicalize \
  -o mlir-vta/test/Target/_spike_bufferized.mlir && echo "PIPE OK"
grep -q "linalg.matmul" mlir-vta/test/Target/_spike_bufferized.mlir && echo "HAS MATMUL"
grep -q "memref" mlir-vta/test/Target/_spike_bufferized.mlir && echo "HAS MEMREF"
```
Expected: `PIPE OK` + `HAS MATMUL` + `HAS MEMREF`，且 `_spike_bufferized.mlir` 里存在一个操作数为 `memref<16x16xi32>` 的 `linalg.matmul`。

- [ ] **Step 4：若方案 2 失败，回退方案 1（先 bufferize 再在 memref 上 tile）**

Run（仅当 Step 3 不满足时）：
```bash
/usr/local/llvm/bin/mlir-opt mlir-vta/test/Target/matmul_tensor.mlir \
  --linalg-bufferize --tensor-bufferize --tensor-constant-bufferize \
  --func-bufferize --finalizing-bufferize \
  --linalg-tile="linalg-tile-sizes=16,16,16" \
  --canonicalize \
  -o mlir-vta/test/Target/_spike_bufferized.mlir && echo "PIPE OK (fallback)"
grep -q "linalg.matmul" mlir-vta/test/Target/_spike_bufferized.mlir && echo "HAS MATMUL"
```
Expected: 二选一能产出含 memref 语义 `linalg.matmul` 的 IR。

- [ ] **Step 5：记录决策到 `_spike_bufferize_notes.md`**

写入：所选方案（2 或回退 1）、**最终可用的完整 pass 序列字符串**（逐字）、`_spike_bufferized.mlir` 中 `linalg.matmul` 的实际形态（operand 类型、是否被 `scf.for` 包裹）。后续 Task 2/4 直接引用此序列。

- [ ] **Step 6：提交**
```bash
printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' \
  'git add test/Target/matmul_tensor.mlir test/Target/_spike_bufferized.mlir docs/superpowers/plans/_spike_bufferize_notes.md' \
  'git commit -m "chore(mlir-vta): phase2 pipeline spike (tile+bufferize) notes"' \
  > /tmp/do_commit.sh && bash /tmp/do_commit.sh
```

---

### Task 1：扩展 `vta.gemm` 承载 memref operand + 适配 lowering

**Files:**
- Modify: `mlir-vta/include/mlir-vta/Dialect/VTA/VTAOps.td`
- Modify: `mlir-vta/lib/Transforms/LowerVTAGemm.cpp`
- Modify: `mlir-vta/test/Target/lower_gemm.mlir`

- [ ] **Step 1：改 `VTAOps.td` 的 `vta.gemm` 定义**

把现有 `VTA_GemmOp`（`ins I64Attr m,n,k`）整体替换为：
```tablegen
def VTA_GemmOp : VTA_Op<"gemm"> {
  let summary = "single 16x16 block GEMM: acc += lhs * rhs (in-place on acc memref)";
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$acc);
  let results = (outs);
  let assemblyFormat =
    "`ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)` "
    "`outs` `(` $acc `:` type($acc) `)` attr-dict";
  let verifier = [{
    auto isBlk = [](::mlir::Value v) {
      auto mt = v.getType().dyn_cast<::mlir::MemRefType>();
      return mt && mt.getRank() == 2 && mt.getShape()[0] == 16 &&
             mt.getShape()[1] == 16;
    };
    if (!isBlk(lhs()) || !isBlk(rhs()) || !isBlk(acc()))
      return emitOpError("phase-2 vta.gemm requires 16x16 memref operands");
    return ::mlir::success();
  }];
}
```
> `AnyMemRef` 由 `mlir/IR/OpBase.td` 提供（经 `VTADialect.td` → `OpBase.td` 传递引入）。LLVM 13 用 `let verifier = [{...}]`（非 `hasVerifier`）。

- [ ] **Step 2：改 `LowerVTAGemm.cpp` 的尺寸守卫（读 memref 形状而非属性）**

把现有守卫块：
```cpp
      // Phase 1 supports only a single 16x16x16 block.
      if (g.m() != 16 || g.n() != 16 || g.k() != 16) {
        g.emitError("lower-vta-gemm: only m=n=k=16 supported in phase 1");
        signalPassFailure();
        return;
      }
```
替换为：
```cpp
      // Phase 2: still only a single 16x16x16 block; derive dims from operands.
      auto is16 = [](Value v) {
        auto mt = v.getType().dyn_cast<MemRefType>();
        return mt && mt.getRank() == 2 && mt.getShape()[0] == 16 &&
               mt.getShape()[1] == 16;
      };
      if (!is16(g.lhs()) || !is16(g.rhs()) || !is16(g.acc())) {
        g.emitError("lower-vta-gemm: only 16x16 memref operands supported");
        signalPassFailure();
        return;
      }
```
> `MemRefType`/`Value` 已由 `VTAOps.h`（含 `BuiltinTypes.h`）引入；`g.erase()` 仍有效（`vta.gemm` 无 result）。其余 11 条 `vtaisa` op 的硬编码生成**完全不变**。

- [ ] **Step 3：更新 round-trip 测试 `lower_gemm.mlir`（高层入口现在带 operand）**

整文件替换为：
```mlir
func @main(%A: memref<16x16xi32>, %B: memref<16x16xi32>, %C: memref<16x16xi32>) {
  "vta.gemm"(%A, %B, %C) : (memref<16x16xi32>, memref<16x16xi32>, memref<16x16xi32>) -> ()
  return
}
```

- [ ] **Step 4：构建**

Run:
```bash
cmake --build ~/mlir-vta-build -j"$(nproc)" --target vta-opt vta-translate
```
Expected: 构建成功（TableGen 重新生成 `VTAOps.*.inc`）。

- [ ] **Step 5：round-trip + lowering 断言**

Run:
```bash
~/mlir-vta-build/bin/vta-opt mlir-vta/test/Target/lower_gemm.mlir | grep -q "vta.gemm" && echo "PARSE+ROUNDTRIP OK"
~/mlir-vta-build/bin/vta-opt -lower-vta-gemm mlir-vta/test/Target/lower_gemm.mlir | grep -q "vtaisa.finish" && echo "LOWER OK"
~/mlir-vta-build/bin/vta-opt -lower-vta-gemm mlir-vta/test/Target/lower_gemm.mlir | grep -q "vta.gemm" && echo "STILL HAS GEMM (BAD)" || echo "GEMM ERASED OK"
```
Expected: `PARSE+ROUNDTRIP OK`、`LOWER OK`、`GEMM ERASED OK`。

- [ ] **Step 6：提交**
```bash
printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' \
  'git add -A' 'git commit -m "feat(mlir-vta): vta.gemm carries 16x16 memref operands"' \
  > /tmp/do_commit.sh && bash /tmp/do_commit.sh
```

---

### Task 2：新 `--convert-linalg-to-vta` pass + `vta-opt` 注册全部上游 pass

**Files:**
- Create: `mlir-vta/lib/Transforms/ConvertLinalgToVTA.cpp`
- Modify: `mlir-vta/include/mlir-vta/Dialect/VTA/VTAPasses.h`
- Modify: `mlir-vta/lib/Transforms/CMakeLists.txt`
- Modify: `mlir-vta/tools/vta-opt/vta-opt.cpp`
- Modify: `mlir-vta/tools/vta-opt/CMakeLists.txt`

- [ ] **Step 1：写 `ConvertLinalgToVTA.cpp`**

```cpp
#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir-vta/Dialect/VTA/VTAOps.h"
#include "mlir-vta/Dialect/VTA/VTAPasses.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct ConvertLinalgToVTAPass
    : public PassWrapper<ConvertLinalgToVTAPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "convert-linalg-to-vta"; }
  StringRef getDescription() const final {
    return "Convert bufferized 16x16 linalg.matmul to vta.gemm";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vta::VTADialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<linalg::MatmulOp> targets;
    module.walk([&](linalg::MatmulOp m) { targets.push_back(m); });

    for (linalg::MatmulOp m : targets) {
      if (!m.hasBufferSemantics()) {
        m.emitError("convert-linalg-to-vta: expected bufferized (memref) "
                    "linalg.matmul; run bufferization first");
        signalPassFailure();
        return;
      }
      Value lhs = m.inputs()[0];
      Value rhs = m.inputs()[1];
      Value acc = m.outputs()[0];

      OpBuilder b(m);
      b.create<vta::GemmOp>(m.getLoc(), lhs, rhs, acc);
      m.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vta::createConvertLinalgToVTAPass() {
  return std::make_unique<ConvertLinalgToVTAPass>();
}
```
> LLVM 13：`linalg::MatmulOp` 在 `mlir/Dialect/Linalg/IR/LinalgOps.h`；`inputs()`/`outputs()` 返回 `ValueRange`；`hasBufferSemantics()` 为 LinalgOp 接口方法。

- [ ] **Step 2：在 `VTAPasses.h` 声明并注册新 pass**

把现有内容替换为：
```cpp
#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace vta {

std::unique_ptr<mlir::Pass> createLowerVTAGemmPass();
std::unique_ptr<mlir::Pass> createConvertLinalgToVTAPass();

void registerVTAPasses();

} // namespace vta
} // namespace mlir
```
并在 `lib/Transforms/LowerVTAGemm.cpp` 末尾的 `registerVTAPasses()` 里追加注册（保留原有 `PassRegistration<LowerVTAGemmPass>()`）：
```cpp
void mlir::vta::registerVTAPasses() {
  PassRegistration<LowerVTAGemmPass>();
  ::mlir::registerPass([]() -> std::unique_ptr<Pass> {
    return mlir::vta::createConvertLinalgToVTAPass();
  });
}
```
> 用 `registerPass(factory)` 形式注册另一个文件里的 pass，避免把 `ConvertLinalgToVTAPass` 类型暴露到本文件。LLVM 13 提供 `registerPass(const PassAllocatorFunction&)`。

- [ ] **Step 3：改 `lib/Transforms/CMakeLists.txt`**

整文件替换为：
```cmake
add_mlir_library(MLIRVTATransforms
  LowerVTAGemm.cpp
  ConvertLinalgToVTA.cpp
  DEPENDS VTAOpsIncGen VTAISAOpsIncGen
  LINK_LIBS PUBLIC MLIRVTA MLIRVTAISA MLIRIR MLIRPass MLIRLinalg)
```

- [ ] **Step 4：改 `tools/vta-opt/vta-opt.cpp`（注册全部上游 dialect/pass）**

整文件替换为：
```cpp
#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir-vta/Dialect/VTA/VTAPasses.h"
#include "mlir-vta/Dialect/VTAISA/VTAISADialect.h"

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  registry.insert<mlir::vta::VTADialect, mlir::vtaisa::VTAISADialect>();
  mlir::vta::registerVTAPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "vta-opt\n", registry));
}
```

- [ ] **Step 5：改 `tools/vta-opt/CMakeLists.txt`（链接全部 dialect/conversion 库）**

整文件替换为：
```cmake
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(vta-opt vta-opt.cpp)
llvm_update_compile_flags(vta-opt)
target_link_libraries(vta-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRVTA MLIRVTAISA MLIRVTATransforms
  MLIROptLib MLIRIR MLIRParser MLIRSupport)
mlir_check_all_link_libraries(vta-opt)
```

- [ ] **Step 6：重新配置 + 构建（改了 CMake）**

Run:
```bash
cmake -S /mnt/c/MLIR-VTA/mlir-vta -B ~/mlir-vta-build -G "Unix Makefiles" \
  -DMLIR_DIR=/usr/local/llvm/lib/cmake/mlir -DLLVM_DIR=/usr/local/llvm/lib/cmake/llvm \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_BUILD_TYPE=Release
cmake --build ~/mlir-vta-build -j"$(nproc)" --target vta-opt
```
Expected: 构建成功，`vta-opt --help` 现在能看到 `-linalg-tile`、`-linalg-bufferize`、`-convert-linalg-to-vta`、`-canonicalize`、`-lower-vta-gemm`。

- [ ] **Step 7：convert pass 单元断言（用 spike 产出的 bufferized IR）**

Run:
```bash
~/mlir-vta-build/bin/vta-opt -convert-linalg-to-vta \
  mlir-vta/test/Target/_spike_bufferized.mlir | tee /tmp/converted.mlir >/dev/null
grep -q "vta.gemm" /tmp/converted.mlir && echo "HAS vta.gemm"
grep -q "linalg.matmul" /tmp/converted.mlir && echo "STILL HAS matmul (BAD)" || echo "matmul ERASED OK"
```
Expected: `HAS vta.gemm` + `matmul ERASED OK`。

- [ ] **Step 8：提交**
```bash
printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' \
  'git add -A' 'git commit -m "feat(mlir-vta): add convert-linalg-to-vta pass; vta-opt registers upstream passes"' \
  > /tmp/do_commit.sh && bash /tmp/do_commit.sh
```

---

### Task 3：二进制发射器改递归遍历（容忍 op 嵌在 func/scf 内）

**Files:**
- Modify: `mlir-vta/lib/Target/VTABinaryEmitter.cpp`
- Create: `mlir-vta/test/Target/gemm16x16_func.mlir`

- [ ] **Step 1：把 `emitBinary` 的顶层 for 循环改为递归 `module.walk`**

在 `VTABinaryEmitter.cpp` 中，把：
```cpp
  for (Operation &op : module.getBody()->getOperations()) {
```
改为（仅改循环外壳与错误返回方式，循环体内部各分支逻辑**保持不变**，但把两处 `return failure();` 改为下面的中断写法）：
```cpp
  LogicalResult walkResult = success();
  module.walk([&](Operation *opPtr) -> WalkResult {
    Operation &op = *opPtr;
```
并把循环体内 `vta.uop_table` 长度不等时的 `return failure();` 改为：
```cpp
        walkResult = failure();
        return WalkResult::interrupt();
```
把未处理 vta/vtaisa op 分支的 `return failure();` 改为：
```cpp
      walkResult = failure();
      return WalkResult::interrupt();
```
在循环体（lambda）末尾、原 `// Other ops are ignored.` 处返回 advance，并闭合 lambda：
```cpp
    return WalkResult::advance();
  });
  if (failed(walkResult))
    return failure();
```
> `module.walk` 默认 post-order；因 `vtaisa.*` 与 `uop_table` 均为叶子 op（无嵌套 region），其相对顺序等于块内文本序，发射顺序正确。`func.func`/`func.return`/`module`/`arith.*`/`memref.*` 等非 vta/vtaisa op 落入「忽略」或「终结符跳过」分支。`WalkResult`/`WalkOrder` 由 `mlir/IR/OpDefinition.h`→`Visitors.h` 引入，无需额外 include。

- [ ] **Step 2：写 func 包裹的发射测试 `gemm16x16_func.mlir`**

把第一阶段 `test/Target/gemm16x16.mlir` 里的 13 条 op（1 条 `vtaisa.uop_table` + 11 条指令 + 1 条 `vtaisa.finish`）原样放进一个 `func`：
```mlir
func @main() {
  "vtaisa.uop_table"() {dst = [0, 0], src = [0, 0], wgt = [0, 0]} : () -> ()
  // ... 此处逐条粘贴 gemm16x16.mlir 中 11 条 vtaisa.load/store/gemm_insn 指令（属性照抄）...
  "vtaisa.finish"() : () -> ()
  return
}
```
> 具体 11 条指令的属性请逐条复制 `test/Target/gemm16x16.mlir` 的现有内容，仅把它们移入 `func @main() { ... return }`。

- [ ] **Step 3：构建**

Run:
```bash
cmake --build ~/mlir-vta-build -j"$(nproc)" --target vta-translate
```
Expected: 构建成功。

- [ ] **Step 4：递归遍历的字节级断言（func 包裹仍应等于 golden）**

Run:
```bash
rm -rf /tmp/vtaout_func && mkdir -p /tmp/vtaout_func
~/mlir-vta-build/bin/vta-translate mlir-vta/test/Target/gemm16x16_func.mlir -o /tmp/vtaout_func
cmp /tmp/vtaout_func/instructions.bin mlir-vta/test/golden/instructions.bin && echo "FUNC INSN OK"
cmp /tmp/vtaout_func/uop.bin mlir-vta/test/golden/uop.bin && echo "FUNC UOP OK"
```
Expected: `FUNC INSN OK` + `FUNC UOP OK`（证明嵌在 func 内的 vtaisa op 被正确递归发射）。

- [ ] **Step 5：回归 —— 顶层（非 func）旧用例仍然正确**

Run:
```bash
rm -rf /tmp/vtaout_top && mkdir -p /tmp/vtaout_top
~/mlir-vta-build/bin/vta-translate mlir-vta/test/Target/gemm16x16.mlir -o /tmp/vtaout_top
cmp /tmp/vtaout_top/instructions.bin mlir-vta/test/golden/instructions.bin && echo "TOP INSN OK"
```
Expected: `TOP INSN OK`（递归遍历对顶层 module body 也兼容）。

- [ ] **Step 6：提交**
```bash
printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' \
  'git add -A' 'git commit -m "feat(mlir-vta): emitter walks recursively to handle ops nested in func/scf"' \
  > /tmp/do_commit.sh && bash /tmp/do_commit.sh
```

---

### Task 4：端到端 —— `linalg.matmul`(tensor) → … → FSIM（验收门）

**Files:**
- Create: `mlir-vta/scripts/run_fsim_linalg.sh`

> 入口 `test/Target/matmul_tensor.mlir` 已在 Task 0 创建。下面的「PIPELINE」占位用 Task 0 在 `_spike_bufferize_notes.md` 里记录的**最终可用 pass 序列**（方案 2 或回退方案 1）替换。

- [ ] **Step 1：手验完整管道能产出可发射的 `vtaisa` 程序**

Run（PIPELINE = Task 0 记录的序列；下面给方案 2 的默认形态）：
```bash
~/mlir-vta-build/bin/vta-opt mlir-vta/test/Target/matmul_tensor.mlir \
  --linalg-tile="linalg-tile-sizes=16,16,16" \
  --linalg-bufferize --tensor-bufferize --tensor-constant-bufferize \
  --func-bufferize --finalizing-bufferize \
  --convert-linalg-to-vta --canonicalize --lower-vta-gemm \
  -o /tmp/lowered_tensor.mlir && echo "PIPE OK"
grep -q "vtaisa.finish" /tmp/lowered_tensor.mlir && echo "HAS vtaisa"
grep -q "linalg.matmul" /tmp/lowered_tensor.mlir && echo "RESIDUAL matmul (BAD)" || echo "no residual matmul"
grep -q "vta.gemm" /tmp/lowered_tensor.mlir && echo "RESIDUAL vta.gemm (BAD)" || echo "no residual vta.gemm"
```
Expected: `PIPE OK` + `HAS vtaisa` + `no residual matmul` + `no residual vta.gemm`。

- [ ] **Step 2：translate 产出二进制，并对 golden 做非阻断 `cmp`**

Run:
```bash
rm -rf /tmp/vtaout_lin && mkdir -p /tmp/vtaout_lin
~/mlir-vta-build/bin/vta-translate /tmp/lowered_tensor.mlir -o /tmp/vtaout_lin \
  --emit-data --input mlir-vta/test/golden/input_16x16.bin \
  --weight mlir-vta/test/golden/weight_16x16.bin
cmp /tmp/vtaout_lin/instructions.bin mlir-vta/test/golden/instructions.bin \
  && echo "INSN MATCHES GOLDEN (信号)" || echo "INSN DIFFERS (非阻断：仅 FSIM 为准)"
cmp /tmp/vtaout_lin/uop.bin mlir-vta/test/golden/uop.bin \
  && echo "UOP MATCHES GOLDEN (信号)" || echo "UOP DIFFERS (非阻断)"
```
Expected: 因 lowering/数据未变，理应打印两个 `MATCHES GOLDEN`（若不一致，仅记录，验收以 Step 4 的 FSIM 为准）。

- [ ] **Step 3：写 `scripts/run_fsim_linalg.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
OUT=/tmp/vtaout_fsim_linalg
OPT=$HOME/mlir-vta-build/bin/vta-opt
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate

rm -rf "$OUT" && mkdir -p "$OUT"
# 1. tensor 入口 → 完整管道 → vtaisa 程序
#    PIPELINE 与 _spike_bufferize_notes.md 一致（方案 2 默认形态）
"$OPT" "$MLIRVTA/test/Target/matmul_tensor.mlir" \
  --linalg-tile="linalg-tile-sizes=16,16,16" \
  --linalg-bufferize --tensor-bufferize --tensor-constant-bufferize \
  --func-bufferize --finalizing-bufferize \
  --convert-linalg-to-vta --canonicalize --lower-vta-gemm \
  -o "$OUT/lowered.mlir"
# 2. 发射二进制 + 数据/CSV
"$TRANSLATE" "$OUT/lowered.mlir" -o "$OUT" \
  --emit-data --input "$MLIRVTA/test/golden/input_16x16.bin" \
  --weight "$MLIRVTA/test/golden/weight_16x16.bin"
# 3. stage 进 standalone-vta 并跑 FSIM
cp "$OUT"/instructions.bin "$OUT"/uop.bin "$OUT"/input.bin "$OUT"/weight.bin \
   "$OUT"/accumulator.bin "$OUT"/out_init.bin "$OUT"/metadata.csv \
   "$OUT"/memory_addresses.csv "$OUT"/layers_name.csv "$SVTA/compiler_output/"
: > "$SVTA/compiler_output/add_accumulator.bin"
cd "$SVTA/examples"
make fsim_compile_single_layer
make fsim_single_layer
echo "===== FSIM REPORT (tail) ====="
tail -40 "$SVTA/log_output/fsim_report.txt"
```

- [ ] **Step 4：跑 FSIM 并与第一阶段结果矩阵逐元素对照（验收门）**

Run:
```bash
chmod +x mlir-vta/scripts/run_fsim_linalg.sh
# 先存第一阶段基线
mlir-vta/scripts/run_fsim.sh > /tmp/fsim_phase1.txt 2>&1
tail -40 /tmp/fsim_phase1.txt | sed -n '/=====/,$p' > /tmp/fsim_phase1_matrix.txt
# 再跑新链路
mlir-vta/scripts/run_fsim_linalg.sh > /tmp/fsim_phase2.txt 2>&1
tail -40 /tmp/fsim_phase2.txt | sed -n '/=====/,$p' > /tmp/fsim_phase2_matrix.txt
diff /tmp/fsim_phase1_matrix.txt /tmp/fsim_phase2_matrix.txt && echo "FSIM RESULT MATCHES PHASE 1 (ACCEPT)"
```
Expected: `diff` 无输出 + 打印 `FSIM RESULT MATCHES PHASE 1 (ACCEPT)`。这是第二阶段的核心验收点。

- [ ] **Step 5：提交**
```bash
printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' \
  'git add -A' 'git commit -m "test(mlir-vta): linalg-entry end-to-end pipeline + FSIM acceptance"' \
  > /tmp/do_commit.sh && bash /tmp/do_commit.sh
```

---

### Task 5：更新设计文档

**Files:**
- Modify: `mlir-vta/docs/DESIGN_cn.md`

- [ ] **Step 1：更新 §2.1 阶段表与 §10.1**

把阶段二状态由「规划中」改为「进行中（linalg 入口·16×16 单块已落地）」。在 §10.1 追加一段说明：
- 已落地：`vta.gemm` 带 memref operand；`--convert-linalg-to-vta`；`vta-opt` 注册全部上游 pass；发射器递归遍历；端到端 `linalg.matmul`(tensor) → tile → bufferize → vta.gemm → lower → translate → FSIM。
- 数值约定边界（spec §8）：本阶段 `linalg.matmul` 为「被识别标记」，数学约定与磁盘转置的对应关系留给通用化阶段。
- 记录 Task 0 spike 的最终 pass 序列与方案（2 或回退 1）。
- 仍未做：多块/通用 `m,n,k`、地址分配 pass、依赖信号量推导、ALU lowering。

- [ ] **Step 2：更新 §11 技术债表**

把「`LowerVTAGemm` 仅支持 16×16×16」「数据发射器仅单块 16×16」两行的「计划」列标注为「通用化阶段（阶段二后续）」，并新增一行：「`vta.gemm` 的 m,n,k 来自 operand 形状，verifier 仅认 16×16 → 通用化阶段放开」。

- [ ] **Step 3：提交**
```bash
printf '%s\n' '#!/usr/bin/env bash' 'set -e' 'cd /mnt/c/MLIR-VTA/mlir-vta' \
  'git add -A' 'git commit -m "docs(mlir-vta): update DESIGN for phase-2 linalg entry"' \
  > /tmp/do_commit.sh && bash /tmp/do_commit.sh
```

---

## 自检清单（实现完成后逐项核对）

1. **Spec 覆盖**：spec §3 管道→Task 0/4；§5 op 扩展→Task 1；§6 convert pass→Task 2；§9 emitter 递归→Task 3；§10 验证/验收→Task 4；§13 文档→Task 5。✅
2. **类型/命名一致**：`vta.gemm` operand 顺序 `(lhs, rhs, acc)` 在 td/verifier/LowerVTAGemm/ConvertLinalgToVTA 四处一致；pass 名 `convert-linalg-to-vta`、`lower-vta-gemm` 全程一致。
3. **无占位符**：除 Task 4 的 PIPELINE 显式引用 Task 0 spike 记录（设计使然）外，其余步骤均含可直接执行的代码/命令。
4. **回退路径**：Task 0 内置方案 2 失败 → 方案 1 的回退，且决策落档。
5. **ABI/磁盘**：`clang++` 构建，build 在 `~/mlir-vta-build`。
6. **Git 坑**：所有提交走临时脚本规避 `--trailer` 注入。

## 已知风险与缓解

| 风险 | 缓解 |
|------|------|
| LLVM 13 tile-on-tensors / slice bufferization 不成熟 | Task 0 spike 先验证；失败回退方案 1（先 bufferize 再 memref 上 tile）并落档 |
| `func` 返回 tensor 的 bufferization 边角 | spike 一并验证；额外 `--tensor-constant-bufferize` 已纳入；最终以 spike 记录的序列为准 |
| `registerAllPasses` 拉入大量库使 `vta-opt` 链接变慢/变大 | 接受（仅工具体积）；构建仍走增量 |
| convert 后残留 `scf.for`(trip-1) 使 vta.gemm 嵌套 | `--canonicalize` 折叠 + Task 3 发射器递归遍历双保险 |
| `linalg::MatmulOp` 的 `hasBufferSemantics()`/`inputs()` 在 13 的具体签名 | 若编译报错，按 `/usr/local/llvm/include/mlir/Dialect/Linalg/IR/LinalgOps.h` 实际接口微调（如 `getInputOperand`/`getInputs`）|
