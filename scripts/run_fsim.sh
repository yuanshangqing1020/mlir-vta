#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
OUT=/tmp/vtaout_fsim
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate

rm -rf "$OUT" && mkdir -p "$OUT"
# 1. Emit all artifacts from mlir-vta
"$TRANSLATE" "$MLIRVTA/test/Target/gemm16x16.mlir" -o "$OUT" \
  --emit-data --input "$MLIRVTA/test/golden/input_16x16.bin" \
  --weight "$MLIRVTA/test/golden/weight_16x16.bin"
# 2. Stage into standalone-vta/compiler_output (FSIM reads from there)
cp "$OUT"/instructions.bin "$OUT"/uop.bin "$OUT"/input.bin "$OUT"/weight.bin \
   "$OUT"/accumulator.bin "$OUT"/out_init.bin "$OUT"/metadata.csv \
   "$OUT"/memory_addresses.csv "$OUT"/layers_name.csv "$SVTA/compiler_output/"
: > "$SVTA/compiler_output/add_accumulator.bin"   # empty unused channel
# 3. Build + run FSIM
cd "$SVTA/examples"
make fsim_compile_single_layer
make fsim_single_layer
echo "===== FSIM REPORT (tail) ====="
tail -40 "$SVTA/log_output/fsim_report.txt"
