#!/usr/bin/env bash
# MaxPool 36x16 → vta.maxpool → byte-level vs upstream golden + FSIM.
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
OPT=$HOME/mlir-vta-build/bin/vta-opt
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate
PY=/home/john/miniconda3/envs/standalone-vta/bin/python3
BR=/tmp/onnx_maxpool
OUT=/tmp/onnx_maxpool_out
GD="$MLIRVTA/test/golden/maxpool_36x16"
CO=$SVTA/compiler_output

rm -rf "$BR" "$OUT" && mkdir -p "$BR" "$OUT"

"$PY" "$MLIRVTA/scripts/onnx_maxpool_bridge.py" -o "$BR"

"$OPT" "$BR/maxpool.mlir" --vta-dram-allocation --lower-vta-maxpool -o "$OUT/lowered.mlir"
"$TRANSLATE" "$OUT/lowered.mlir" -o "$OUT"

if [[ ! -d "$GD" || ! -f "$GD/instructions.bin" ]]; then
  bash "$MLIRVTA/scripts/make_golden_maxpool_upstream.sh"
fi

echo "=== byte-level vs upstream golden ==="
cmp "$OUT/instructions_T3.bin" "$GD/instructions.bin" && echo "INSN OK"
cmp "$OUT/uop_T3.bin" "$GD/uop.bin" && echo "UOP OK"

ACC_BLOCKED=$($PY -c "import json; print(json.load(open('$BR/bridge_meta.json'))['paths']['accumulator_blocked'])")
cmp -n "$(stat -c%s "$GD/accumulator_raw.bin")" \
  "$ACC_BLOCKED" "$GD/accumulator_raw.bin" && echo "ACC RAW OK"

cp "$OUT/instructions_T3.bin" "$OUT/uop_T3.bin" "$ACC_BLOCKED" \
   "$OUT/metadata_T3.csv" "$OUT/memory_addresses_T3.csv" "$CO/" 2>/dev/null || true
cp "$ACC_BLOCKED" "$CO/accumulator.bin"
: > "$CO/add_accumulator.bin"
: > "$CO/input.bin"
: > "$CO/weight.bin"

rm -f "$SVTA/log_output/fsim_report.txt"
cd "$SVTA/examples"
make fsim_compile_single_layer >/dev/null 2>&1
cp "$CO/instructions_T3.bin" "$CO/instructions.bin"
cp "$CO/uop_T3.bin" "$CO/uop.bin"
make fsim_single_layer >/dev/null 2>&1
sed -n '/Final result/,/^} $/p' "$SVTA/log_output/fsim_report.txt" > "$OUT/fsim_result.txt"

if [[ -f "$GD/fsim_result.txt" ]]; then
  diff -q "$OUT/fsim_result.txt" "$GD/fsim_result.txt" && echo "=== FSIM vs UPSTREAM: ACCEPT ==="
fi
echo "Increment E (MaxPool 36x16) complete."
