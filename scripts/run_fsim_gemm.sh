#!/usr/bin/env bash
# End-to-end acceptance for a pure GEMM C[MxN] = A[MxK] * B[KxN] of arbitrary
# 16-multiple dims: tensor linalg.matmul -> bufferize -> vta.gemm -> generalized
# lower -> translate (multi-block data) -> FSIM, then check the result matrix
# against the upstream golden test/golden/matmul_<M>x<K>x<N>/.
# Usage: scripts/run_fsim_gemm.sh <M> <K> <N>
set -euo pipefail
M=${1:?M}; K=${2:?K}; N=${3:?N}
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
OPT=$HOME/mlir-vta-build/bin/vta-opt
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate
TAG="${M}x${K}x${N}"
GD="$MLIRVTA/test/golden/matmul_${TAG}"
OUT="/tmp/vtaout_fsim_${TAG}"
rm -rf "$OUT" && mkdir -p "$OUT"

# tensor-level entry (no linalg-tile; block schedule lives in lower-vta-gemm)
cat > "$OUT/entry.mlir" <<EOF
func @main(%A: tensor<${M}x${K}xi32>, %B: tensor<${K}x${N}xi32>) -> tensor<${M}x${N}xi32> {
  %c0 = constant 0 : i32
  %init = linalg.init_tensor [${M}, ${N}] : tensor<${M}x${N}xi32>
  %acc = linalg.fill(%c0, %init) : i32, tensor<${M}x${N}xi32> -> tensor<${M}x${N}xi32>
  %out = linalg.matmul ins(%A, %B : tensor<${M}x${K}xi32>, tensor<${K}x${N}xi32>)
                       outs(%acc : tensor<${M}x${N}xi32>) -> tensor<${M}x${N}xi32>
  return %out : tensor<${M}x${N}xi32>
}
EOF

"$OPT" "$OUT/entry.mlir" \
  --linalg-bufferize --tensor-bufferize --tensor-constant-bufferize \
  --func-bufferize --finalizing-bufferize \
  --convert-linalg-to-vta --canonicalize --lower-vta-gemm \
  -o "$OUT/lowered.mlir"
"$TRANSLATE" "$OUT/lowered.mlir" -o "$OUT" \
  --emit-data --rows "$M" --cols "$N" --k "$K" \
  --input "$GD/input_${M}x${K}.bin" --weight "$GD/weight_${K}x${N}.bin"

cp "$OUT"/instructions.bin "$OUT"/uop.bin "$OUT"/input.bin "$OUT"/weight.bin \
   "$OUT"/accumulator.bin "$OUT"/out_init.bin "$OUT"/metadata.csv \
   "$OUT"/memory_addresses.csv "$OUT"/layers_name.csv "$SVTA/compiler_output/"
: > "$SVTA/compiler_output/add_accumulator.bin"
rm -f "$SVTA/log_output/fsim_report.txt"
cd "$SVTA/examples"
make fsim_compile_single_layer >/dev/null
make fsim_single_layer >/dev/null
echo "===== FSIM profiler ($TAG) ====="
sed -n '/Profiler Status/,/}/p' "$SVTA/log_output/fsim_report.txt"

sed -n '/Final result/,/^} $/p' "$SVTA/log_output/fsim_report.txt" > "$OUT/fsim_result.txt"
if diff -q "$OUT/fsim_result.txt" "$GD/fsim_result_${TAG}.txt" >/dev/null; then
  echo "FSIM RESULT MATCHES GOLDEN ${TAG} (ACCEPT)"
else
  echo "FSIM RESULT DIFFERS FROM GOLDEN ${TAG} (REJECT)" >&2
  diff "$GD/fsim_result_${TAG}.txt" "$OUT/fsim_result.txt" >&2 || true
  exit 1
fi
