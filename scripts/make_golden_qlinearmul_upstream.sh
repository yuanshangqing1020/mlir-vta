#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
GD="$MLIRVTA/test/golden/matmulConst_16x16"
CO=$SVTA/compiler_output
PY=/home/john/miniconda3/envs/standalone-vta/bin/python3
JSON=$SVTA/examples/vta_ir/matmulConst_16x16.json
STAGE="$MLIRVTA/scripts/stage_fsim_single_layer.sh"

mkdir -p "$GD" "$CO"
"$PY" "$MLIRVTA/scripts/onnx_qlinearmul_bridge.py" -o /tmp/qlinearmul_golden_gen
cp /tmp/qlinearmul_golden_gen/input__T2_16x16x16.bin "$CO/input_16x16.bin" 2>/dev/null || \
  cp /tmp/qlinearmul_golden_gen/input*.bin "$CO/input_16x16.bin"

cp "$JSON" "$CO/_T2.json"
"$PY" "$SVTA/src/compiler/vta_compiler/main_vta_compiler.py" False True False \
  "$SVTA/config/vta_config.json" "$CO/_T2.json" >/dev/null

: > "$CO/add_accumulator_T2.bin"
: > "$CO/accumulator_T2.bin"
rm -f "$SVTA/log_output/fsim_report.txt"

# Stage suffixed artifacts for fsim_single_layer (avoid stale layers_name.csv).
TMP=/tmp/qlinearmul_golden_stage
rm -rf "$TMP" && mkdir -p "$TMP"
cp "$CO/instructions_T2.bin" "$TMP/instructions_T2.bin"
cp "$CO/uop_T2.bin" "$TMP/uop_T2.bin"
cp "$CO/input_16x16.bin" "$TMP/input_T2.bin"
cp "$CO/weight_T2.bin" "$TMP/weight_T2.bin"
cp "$CO/metadata_T2.csv" "$TMP/metadata_T2.csv"
cp "$CO/memory_addresses_T2.csv" "$TMP/memory_addresses_T2.csv"
cp "$CO/layers_name.csv" "$TMP/layers_name.csv"
bash "$STAGE" "$TMP" "_T2" "$CO"

FSIM_DIR=$SVTA/src/simulators/functional_simulator
cd "$FSIM_DIR"
make build/fsim_single_layer >/dev/null 2>&1 || true
timeout 120 ./build/fsim_single_layer > "$SVTA/log_output/fsim_report.txt" 2>&1

cp "$CO/instructions_T2.bin" "$GD/instructions.bin"
cp "$CO/uop_T2.bin" "$GD/uop.bin"
cp "$CO/input_16x16.bin" "$GD/input.bin"
cp "$CO/weight_T2.bin" "$GD/weight.bin"
cp "$CO/metadata_T2.csv" "$GD/metadata.csv" 2>/dev/null || true
cp "$CO/memory_addresses_T2.csv" "$GD/memory_addresses.csv" 2>/dev/null || true
sed -n '/Final result/,/^} $/p' "$SVTA/log_output/fsim_report.txt" > "$GD/fsim_result.txt"
echo "Upstream matmulConst golden -> $GD"
ls -la "$GD"
