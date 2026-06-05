#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
GD="$MLIRVTA/test/golden/maxpool_36x16"
CO=$SVTA/compiler_output
PY=/home/john/miniconda3/envs/standalone-vta/bin/python3
JSON=$SVTA/examples/vta_ir/maxpool_36x16.json

mkdir -p "$GD"
"$PY" "$MLIRVTA/scripts/onnx_maxpool_bridge.py" -o /tmp/maxpool_golden_gen
cp /tmp/maxpool_golden_gen/accumulator_36x16.bin "$CO/accumulator_36x16.bin"

cp "$JSON" "$CO/_T3.json"
cd "$SVTA/examples"
$PY ../src/compiler/vta_compiler/main_vta_compiler.py False True False \
  ../config/vta_config.json ../compiler_output/_T3.json >/dev/null

: > "$CO/add_accumulator.bin"
rm -f "$SVTA/log_output/fsim_report.txt"
make fsim_compile_single_layer >/dev/null 2>&1
make fsim_single_layer >/dev/null 2>&1

cp "$CO/instructions_T3.bin" "$GD/instructions.bin"
cp "$CO/uop_T3.bin" "$GD/uop.bin"
cp "$CO/accumulator_36x16.bin" "$GD/accumulator_raw.bin"
cp "$CO/accumulator.bin" "$GD/accumulator.bin" 2>/dev/null || \
  cp "$CO/accumulator_36x16.bin" "$GD/accumulator.bin"
cp "$CO/metadata_T3.csv" "$GD/metadata.csv" 2>/dev/null || true
cp "$CO/memory_addresses_T3.csv" "$GD/memory_addresses.csv" 2>/dev/null || true
sed -n '/Final result/,/^} $/p' "$SVTA/log_output/fsim_report.txt" > "$GD/fsim_result.txt"
echo "Upstream maxpool golden -> $GD"
ls "$GD"
