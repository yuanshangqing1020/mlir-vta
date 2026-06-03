#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
OUT=/tmp/vtaout_fsim_linalg_32x32
OPT=$HOME/mlir-vta-build/bin/vta-opt
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate
GD=$MLIRVTA/test/golden/matmul_32x32

rm -rf "$OUT" && mkdir -p "$OUT"
# 1. tensor 32x32 entry -> full pipeline (NO linalg-tile) -> vtaisa program
"$OPT" "$MLIRVTA/test/Target/matmul_tensor_32x32.mlir" \
  --linalg-bufferize --tensor-bufferize --tensor-constant-bufferize \
  --func-bufferize --finalizing-bufferize \
  --convert-linalg-to-vta --canonicalize --lower-vta-gemm \
  -o "$OUT/lowered.mlir"
# 2. emit instructions/uop + multi-block data/CSV (32x32x32)
"$TRANSLATE" "$OUT/lowered.mlir" -o "$OUT" \
  --emit-data --input "$GD/input_32x32.bin" --weight "$GD/weight_32x32.bin" \
  --rows 32 --cols 32 --k 32
# 3. stage into standalone-vta and run FSIM
cp "$OUT"/instructions.bin "$OUT"/uop.bin "$OUT"/input.bin "$OUT"/weight.bin \
   "$OUT"/accumulator.bin "$OUT"/out_init.bin "$OUT"/metadata.csv \
   "$OUT"/memory_addresses.csv "$OUT"/layers_name.csv "$SVTA/compiler_output/"
: > "$SVTA/compiler_output/add_accumulator.bin"
rm -f "$SVTA/log_output/fsim_report.txt"
cd "$SVTA/examples"
make fsim_compile_single_layer
make fsim_single_layer
echo "===== FSIM REPORT (profiler) ====="
sed -n '/Profiler Status/,/}/p' "$SVTA/log_output/fsim_report.txt"

# 4. acceptance gate: result matrix (4 blocks) == golden, element-wise
REPORT="$SVTA/log_output/fsim_report.txt"
GOLDEN="$GD/fsim_result_32x32.txt"
sed -n '/Final result/,/^} $/p' "$REPORT" > "$OUT/fsim_result.txt"
if [ ! -s "$OUT/fsim_result.txt" ]; then
  echo "FSIM result matrix empty -- extraction failed (REJECT)" >&2; exit 1
fi
if diff -q "$OUT/fsim_result.txt" "$GOLDEN" >/dev/null; then
  echo "FSIM RESULT MATCHES GOLDEN 32x32 (ACCEPT)"
else
  echo "FSIM RESULT DIFFERS FROM GOLDEN 32x32 (REJECT)" >&2
  diff "$GOLDEN" "$OUT/fsim_result.txt" >&2 || true
  exit 1
fi
