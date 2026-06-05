#!/usr/bin/env bash
# Generate byte-level golden for padded qlinearconv_debug.onnx GEMM via upstream VTA compiler.
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
GD="$MLIRVTA/test/golden/qlinearconv_padded"
BR=/tmp/onnx_qconv_bridge_golden
CO=$SVTA/compiler_output
ONNX=$SVTA/examples/onnx/qlinearconv_debug.onnx

rm -rf "$BR" && mkdir -p "$BR" "$GD"
python3 "$MLIRVTA/scripts/onnx_qconv_bridge.py" "$ONNX" -o "$BR"

M=$(python3 -c "import json; d=json.load(open('$BR/bridge_meta.json')); print(d['raw_dims']['M'])")
K=$(python3 -c "import json; d=json.load(open('$BR/bridge_meta.json')); print(d['raw_dims']['K'])")
N=$(python3 -c "import json; d=json.load(open('$BR/bridge_meta.json')); print(d['raw_dims']['N'])")
NAME=$(python3 -c "import json; d=json.load(open('$BR/bridge_meta.json')); print(d['name'])")

JSON="$SVTA/examples/vta_ir/qlinearconv_padded.json"
cat > "$JSON" <<EOF
{
  "NAME": "${NAME}",
  "MATRICES": {
    "A": [${M}, ${K}, "../compiler_output/input_${M}x${K}.bin"],
    "B": [${K}, ${N}, "../compiler_output/weight_${K}x${N}.bin"],
    "X": [${M}, ${N}, "../compiler_output/accumulator_${M}x${N}.bin"],
    "C": [${M}, ${N}, "output"]
  },
  "LOAD": { "INP": ["A"], "WGT": ["B"], "ACC": ["X"] },
  "GEMM": ["C", "A", "B"],
  "STORE": { "C": ["C"] },
  "STRATEGY": 2
}
EOF

cp "$BR/input_${M}x${K}.bin" "$CO/"
cp "$BR/weight_${K}x${N}.bin" "$CO/"
cp "$BR/accumulator_${M}x${N}.bin" "$CO/"
cp "$JSON" "$CO/"

cd "$SVTA/examples"
python3 ../src/compiler/vta_compiler/main_vta_compiler.py False True False \
  ../config/vta_config.json "../compiler_output/qlinearconv_padded.json" >/dev/null

: > "$CO/add_accumulator.bin"
rm -f "$SVTA/log_output/fsim_report.txt"
make fsim_compile_single_layer >/dev/null 2>&1
make fsim_single_layer >/dev/null 2>&1

cp "$CO/instructions${NAME}.bin" "$GD/instructions.bin" 2>/dev/null || \
  cp "$CO/instructions.bin" "$GD/instructions.bin"
cp "$CO/uop${NAME}.bin" "$GD/uop.bin" 2>/dev/null || cp "$CO/uop.bin" "$GD/uop.bin"
cp "$CO/input${NAME}.bin" "$GD/input.bin" 2>/dev/null || cp "$CO/input.bin" "$GD/input.bin"
cp "$CO/weight${NAME}.bin" "$GD/weight.bin" 2>/dev/null || cp "$CO/weight.bin" "$GD/weight.bin"
cp "$CO/accumulator${NAME}.bin" "$GD/accumulator.bin" 2>/dev/null || \
  cp "$CO/accumulator.bin" "$GD/accumulator.bin"
cp "$CO/metadata${NAME}.csv" "$GD/metadata.csv" 2>/dev/null || cp "$CO/metadata.csv" "$GD/metadata.csv"
cp "$CO/memory_addresses${NAME}.csv" "$GD/memory_addresses.csv" 2>/dev/null || \
  cp "$CO/memory_addresses.csv" "$GD/memory_addresses.csv"
cp "$CO/layers_name.csv" "$GD/layers_name.csv"
cp "$BR/input_${M}x${K}.bin" "$GD/input_${M}x${K}.bin"
cp "$BR/weight_${K}x${N}.bin" "$GD/weight_${K}x${N}.bin"
cp "$BR/accumulator_${M}x${N}.bin" "$GD/accumulator_${M}x${N}.bin"
cp "$BR/qlinearconv_padded.mlir" "$MLIRVTA/test/Target/qlinearconv_padded.mlir"
sed -n '/Final result/,/^} $/p' "$SVTA/log_output/fsim_report.txt" > "$GD/fsim_result.txt"

echo "Golden written to $GD (GEMM ${M}x${K}x${N}, strategy 2)"
ls "$GD"
