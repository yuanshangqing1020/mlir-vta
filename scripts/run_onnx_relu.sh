#!/usr/bin/env bash
# Increment E: 16x16 ReLU (MAX_IMM) → vta.alu → byte-level vs upstream golden + FSIM.
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
OPT=$HOME/mlir-vta-build/bin/vta-opt
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate
PY=/home/john/miniconda3/envs/standalone-vta/bin/python3
ONNX=${1:-}
BR=/tmp/onnx_relu
OUT=/tmp/onnx_relu_out
GD="$MLIRVTA/test/golden/relu_16x16"
CO=$SVTA/compiler_output

rm -rf "$BR" "$OUT" && mkdir -p "$BR" "$OUT"

if [[ -n "$ONNX" ]]; then
  "$PY" "$MLIRVTA/scripts/onnx_relu_bridge.py" "$ONNX" -o "$BR"
else
  "$PY" "$MLIRVTA/scripts/onnx_relu_bridge.py" -o "$BR"
fi

M=$($PY -c "import json; print(json.load(open('$BR/bridge_meta.json'))['M'])")
NAME=$($PY -c "import json; print(json.load(open('$BR/bridge_meta.json'))['name'])")
ACC_BLOCKED=$($PY -c "import json; print(json.load(open('$BR/bridge_meta.json'))['paths']['accumulator_blocked'])")

"$OPT" "$BR/relu.mlir" --lower-vta-alu -o "$OUT/lowered.mlir"
"$TRANSLATE" "$OUT/lowered.mlir" -o "$OUT"

if [[ ! -d "$GD" || ! -f "$GD/instructions.bin" ]]; then
  echo "Generating upstream golden..."
  bash "$MLIRVTA/scripts/make_golden_relu_upstream.sh"
fi

echo "=== byte-level vs upstream golden ==="
cmp "$OUT/instructions.bin" "$GD/instructions.bin" && echo "INSN OK"
cmp "$OUT/uop.bin" "$GD/uop.bin" && echo "UOP OK"
cmp -n "$(stat -c%s "$GD/accumulator_raw.bin")" \
  "$ACC_BLOCKED" "$GD/accumulator_raw.bin" && echo "ACC RAW OK"
cmp "$CO/accumulator.bin" "$GD/accumulator.bin" 2>/dev/null && echo "ACC BLOCKED OK" || true

cp "$OUT/instructions.bin" "$OUT/uop.bin" "$ACC_BLOCKED" \
   "$OUT/metadata.csv" "$OUT/memory_addresses.csv" "$OUT/layers_name.csv" "$CO/" 2>/dev/null || true
cp "$ACC_BLOCKED" "$CO/accumulator.bin"
: > "$CO/add_accumulator.bin"
: > "$CO/input.bin"
: > "$CO/weight.bin"

rm -f "$SVTA/log_output/fsim_report.txt"
cd "$SVTA/examples"
make fsim_compile_single_layer >/dev/null 2>&1
make fsim_single_layer >/dev/null 2>&1
sed -n '/Final result/,/^} $/p' "$SVTA/log_output/fsim_report.txt" > "$OUT/fsim_result.txt"

if [[ -f "$GD/fsim_result.txt" ]]; then
  diff -q "$OUT/fsim_result.txt" "$GD/fsim_result.txt" && echo "=== FSIM vs UPSTREAM: ACCEPT ==="
fi

echo "Increment E (ReLU 16x16 MAX_IMM) complete."
