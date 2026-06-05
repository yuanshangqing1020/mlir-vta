#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
GD="$MLIRVTA/test/golden/matmulConst_16x16"
CO=$SVTA/compiler_output
PY=/home/john/miniconda3/envs/standalone-vta/bin/python3
JSON=$SVTA/examples/vta_ir/matmulConst_16x16.json

mkdir -p "$GD"
"$PY" "$MLIRVTA/scripts/onnx_qlinearmul_bridge.py" -o /tmp/qlinearmul_golden_gen
cp /tmp/qlinearmul_golden_gen/input__T2_16x16x16.bin "$CO/input_16x16.bin" 2>/dev/null || \
  cp /tmp/qlinearmul_golden_gen/input*.bin "$CO/input_16x16.bin"

cp "$JSON" "$CO/_T2.json"
cd "$SVTA/examples"
$PY ../src/compiler/vta_compiler/main_vta_compiler.py False True False \
  ../config/vta_config.json ../compiler_output/_T2.json >/dev/null

: > "$CO/add_accumulator.bin"
: > "$CO/accumulator.bin"
rm -f "$SVTA/log_output/fsim_report.txt"
make fsim_compile_single_layer >/dev/null 2>&1
make fsim_single_layer >/dev/null 2>&1

cp "$CO/instructions_T2.bin" "$GD/instructions.bin"
cp "$CO/uop_T2.bin" "$GD/uop.bin"
cp "$CO/input_16x16.bin" "$GD/input.bin"
cp "$CO/weight.bin" "$GD/weight.bin" 2>/dev/null || true
cp "$CO/metadata_T2.csv" "$GD/metadata.csv" 2>/dev/null || true
cp "$CO/memory_addresses_T2.csv" "$GD/memory_addresses.csv" 2>/dev/null || true
sed -n '/Final result/,/^} $/p' "$SVTA/log_output/fsim_report.txt" > "$GD/fsim_result.txt"
echo "Upstream matmulConst golden -> $GD"
ls "$GD"
