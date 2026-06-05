#!/usr/bin/env bash
# Numeric golden for conv_relu_debug.onnx (dependency.csv + NumPy reference, seed=42).
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
GD="$MLIRVTA/test/golden/conv_relu_debug"
CO=$SVTA/compiler_output
PY=/home/john/miniconda3/envs/standalone-vta/bin/python3
ONNX=$SVTA/examples/onnx/conv_relu_debug.onnx
BR=/tmp/conv_relu_golden_gen

mkdir -p "$GD" "$CO"
"$PY" "$MLIRVTA/scripts/gen_conv_relu_debug.py"

rm -rf "$BR" && mkdir -p "$BR"
"$PY" "$MLIRVTA/scripts/onnx_conv_relu_bridge.py" "$ONNX" -o "$BR" --layout col-pad

$PY - <<PY
import json, sys
sys.path.insert(0, "$MLIRVTA/scripts")
from emit_dependency_csv import write_dependency_csv
layers = json.load(open("$BR/bridge_meta.json"))["layers"]
conv = layers[0]
oc, oh, ow = layers[-1]["output_shape"][1:]
write_dependency_csv(
    "$CO/dependency.csv", layers,
    image_rows=conv["raw_dims"]["ah"],
    image_cols=conv["input_shape"][1],
    output_layer=layers[-1]["name"],
    output_nchw=(oc, oh, ow))
print("dependency.csv -> $CO/dependency.csv")
PY

"$PY" "$MLIRVTA/scripts/gen_conv_relu_reference.py" "$ONNX" "$CO/dependency.csv" -o "$GD"

cp "$CO/dependency.csv" "$GD/dependency.csv"
echo "Conv+Relu numeric golden -> $GD"
ls -la "$GD"
