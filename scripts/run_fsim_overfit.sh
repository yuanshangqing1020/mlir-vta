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
# 1. tensor overfit entry (M=16, K=2064, N=16) -> full pipeline (NO linalg-tile) -> vtaisa
"$OPT" "$MLIRVTA/test/Target/matmul_tensor_overfit.mlir" \
  --linalg-bufferize --tensor-bufferize --tensor-constant-bufferize \
  --func-bufferize --finalizing-bufferize \
  --convert-linalg-to-vta --canonicalize --lower-vta-gemm \
  -o "$OUT/lowered.mlir"
# 2. emit instructions/uop + multi-block data/CSV (strategy-1, 2 steps)
"$TRANSLATE" "$OUT/lowered.mlir" -o "$OUT" \
  --emit-data --input "$GD/input_16x2064.bin" --weight "$GD/weight_2064x16.bin" \
  --rows 16 --cols 16 --k 2064
# 3. stage into standalone-vta and run FSIM
cp "$OUT"/instructions.bin "$OUT"/uop.bin "$OUT"/input.bin "$OUT"/weight.bin \
   "$OUT"/accumulator.bin "$OUT"/out_init.bin "$OUT"/metadata.csv \
   "$OUT"/memory_addresses.csv "$OUT"/layers_name.csv "$SVTA/compiler_output/"
: > "$SVTA/compiler_output/add_accumulator.bin"
rm -f "$SVTA/log_output/fsim_report.txt"
cd "$SVTA/examples"
make fsim_compile_single_layer
make fsim_single_layer
echo "===== FSIM REPORT (tail) ====="
tail -20 "$SVTA/log_output/fsim_report.txt"

# 4. acceptance gate: result matrix == golden (upstream Python compiler output)
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
