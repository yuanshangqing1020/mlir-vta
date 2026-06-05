#!/usr/bin/env bash
# Increment F-a: two parallel QLinearMul (VTA) + QLinearAdd (CPU qadd) → fsim_nn.
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
OPT=$HOME/mlir-vta-build/bin/vta-opt
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate
PY=/home/john/miniconda3/envs/standalone-vta/bin/python3
FSIM_DIR=$SVTA/src/simulators/functional_simulator
ONNX=$SVTA/examples/onnx/qadd_two_mul.onnx
GD="$MLIRVTA/test/golden/qadd_two_mul"
BR=/tmp/onnx_qadd_two_mul
OUT=/tmp/onnx_qadd_two_mul_out
CO=$SVTA/compiler_output

if [[ ! -f "$ONNX" ]]; then
  "$PY" "$MLIRVTA/scripts/gen_qadd_two_mul.py"
fi

if [[ ! -f "$GD/reference.bin" ]]; then
  mkdir -p "$GD"
fi

rm -rf "$BR" "$OUT" && mkdir -p "$BR" "$OUT"

"$PY" "$MLIRVTA/scripts/onnx_qadd_two_mul_bridge.py" "$ONNX" -o "$BR"

SHARED_IN=$($PY -c "import json; print(json.load(open('$BR/bridge_meta.json'))['shared_input'])")
mkdir -p "$GD"
"$PY" "$MLIRVTA/scripts/gen_qadd_two_mul_reference.py" \
  --input-matrix "$SHARED_IN" -o "$GD"

"$OPT" "$BR/qadd_two_mul.mlir" --vta-dram-allocation --lower-vta-gemm -o "$OUT/lowered.mlir"

LAYER_ARGS=$($PY - <<PY
import json
layers = json.load(open("$BR/bridge_meta.json"))["layers"]
parts = []
for l in layers:
    if l.get("processor") != "vta":
        continue
    d = l["raw_dims"]
    p = l["paths"]
    parts.append(f'--layer {l["name"]}={d["M"]},{d["K"]},{d["N"]},{p["input"]},{p["weight"]}')
print(" ".join(parts))
PY
)

# shellcheck disable=SC2086
"$TRANSLATE" "$OUT/lowered.mlir" -o "$OUT" $LAYER_ARGS

$PY - <<PY
import json, sys
sys.path.insert(0, "$MLIRVTA/scripts")
from emit_dependency_csv import write_dependency_csv
layers = json.load(open("$BR/bridge_meta.json"))["layers"]
write_dependency_csv(
    "$OUT/dependency.csv", layers,
    image_rows=16, image_cols=16,
    output_layer=layers[-1]["name"],
    output_nchw=(16, 1, 16))
print("dependency.csv written")
PY

cp "$OUT/dependency.csv" "$CO/dependency.csv"
cp "$OUT/layers_name.csv" "$CO/layers_name.csv" 2>/dev/null || true
cp "$GD/input_nn.bin" "$CO/input_nn.bin"

for l in $($PY -c "import json; print(' '.join(x['name'] for x in json.load(open('$BR/bridge_meta.json'))['layers'] if x.get('processor')=='vta'))"); do
  : > "$CO/add_accumulator${l}.bin"
done

for f in "$OUT"/instructions*.bin "$OUT"/uop*.bin "$OUT"/input*.bin "$OUT"/weight*.bin \
         "$OUT"/accumulator*.bin "$OUT"/metadata*.csv "$OUT"/memory_addresses*.csv; do
  [[ -f "$f" ]] && cp "$f" "$CO/"
done

echo "=== Run fsim_nn (2× Mul + QLinearAdd) ==="
cd "$FSIM_DIR"
./build/fsim_nn 2>&1 | grep -E "steps|VTA|written|ERROR|qadd" || true

[[ -f "$CO/final_output.bin" ]] || { echo "ERROR: final_output.bin missing" >&2; exit 1; }

echo "=== Numeric compare vs NumPy reference (seed=42) ==="
cmp "$CO/final_output.bin" "$GD/reference.bin"
echo "=== FSIM vs REFERENCE: ACCEPT (F-a QLinearAdd) ==="
