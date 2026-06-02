# Phase-2 Task 0: Pipeline Spike — tile + bufferization (LLVM 13.0.0)

Goal: lower a tensor-level `linalg.matmul` (16×16) down to a `memref`-semantics
`linalg.matmul` using only upstream `/usr/local/llvm/bin/mlir-opt` (LLVM 13.0.0).
No C++ built or modified.

## Decision: METHOD 2 (tile-on-tensors → bufferize) — WORKS

Method 2 succeeded on the first attempt after fixing the entry-IR syntax. No need to
fall back to Method 1.

## Environment notes / IR syntax fixes

- `mlir-opt --version`: LLVM 13.0.0, x86_64-unknown-linux-gnu.
- The `arith` dialect is NOT registered in this `mlir-opt` build: `arith.constant`
  is rejected ("custom op 'arith.constant' is unknown"). The entry IR uses the
  std-style `constant 0 : i32` instead, which parses.
- `func @main(...)`, `linalg.init_tensor`, and `linalg.fill(%c0, %init) : i32,
  tensor<...> -> tensor<...>` all parse as-is in LLVM 13.0.0 — no other changes needed.
- Tile-size flag: the option is spelled `--linalg-tile-sizes` (NOT a sub-option of a
  bracketed `--linalg-tile=...`). The form `--linalg-tile="linalg-tile-sizes=16,16,16"`
  is accepted by this build and works.

## Final pass sequence (copy-paste runnable, VERBATIM)

```bash
/usr/local/llvm/bin/mlir-opt mlir-vta/test/Target/matmul_tensor.mlir \
  --linalg-tile="linalg-tile-sizes=16,16,16" \
  --linalg-bufferize --tensor-bufferize --tensor-constant-bufferize \
  --func-bufferize --finalizing-bufferize \
  --canonicalize \
  -o mlir-vta/test/Target/_spike_bufferized.mlir
```

(When run from inside the `mlir-vta/` directory, drop the `mlir-vta/` path prefix on
the input/output files, e.g. `test/Target/matmul_tensor.mlir`.)

Verification:
```bash
grep -q "linalg.matmul" mlir-vta/test/Target/_spike_bufferized.mlir && echo "HAS MATMUL"
grep -q "memref"        mlir-vta/test/Target/_spike_bufferized.mlir && echo "HAS MEMREF"
```
Both print → PIPE OK / HAS MATMUL / HAS MEMREF.

## Observed form of the bufferized `linalg.matmul`

From `test/Target/_spike_bufferized.mlir`:

```mlir
module  {
  func @main(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>) -> memref<16x16xi32> {
    %c0_i32 = constant 0 : i32
    %0 = memref.alloc() : memref<16x16xi32>
    linalg.fill(%c0_i32, %0) : i32, memref<16x16xi32>
    %1 = memref.alloc() : memref<16x16xi32>
    linalg.copy(%0, %1) : memref<16x16xi32>, memref<16x16xi32>
    linalg.matmul ins(%arg0, %arg1 : memref<16x16xi32>, memref<16x16xi32>) outs(%1 : memref<16x16xi32>)
    return %1 : memref<16x16xi32>
  }
}
```

Key facts for downstream tasks:

- **Operand types**: all three operands are `memref<16x16xi32>` (fully bufferized,
  static shape, no layout map).
- **inputs/outputs order**: `ins(%A, %B)` then `outs(%acc)` — i.e.
  `ins(lhs, rhs : memref, memref) outs(acc : memref)`. The matmul has NO results
  (memref out-param semantics); the accumulator buffer `%1` is the output.
- **NOT wrapped in `scf.for`**: because tile size (16,16,16) equals the matrix
  dimensions (16×16×16), tiling produces a single trivial tile and `--canonicalize`
  folds away the surrounding loop. The `linalg.matmul` sits directly in the function
  body. (If smaller tiles were requested, an `scf.for` nest would remain — relevant if
  later tasks tile to <16.)
- The fill is done into a separate alloc `%0`, then `linalg.copy`'d into `%1` which
  becomes the matmul accumulator/output. `func @main` signature became
  `(memref, memref) -> memref` after `--func-bufferize`.

## Files

- Input:  `mlir-vta/test/Target/matmul_tensor.mlir` (entry IR, parses with `PARSE OK`)
- Output: `mlir-vta/test/Target/_spike_bufferized.mlir` (spike output; genuine memref
  `linalg.matmul`, usable as test input for later tasks)
