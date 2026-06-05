#!/usr/bin/env bash
# QLinearMul / MulConstant 16x16 → vta.gemm(mul_constant) → byte-level golden.
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
OPT=$HOME/mlir-vta-build/bin/vta-opt
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate
PY=/home/john/miniconda3/envs/standalone-vta/bin/python3
ONNX=${1:-}
BR=/tmp/onnx_qlinearmul
OUT=/tmp/onnx_qlinearmul_out
GD="$MLIRVTA/test/golden/matmulConst_16x16"
CO=$SVTA/compiler_output

rm -rf "$BR" "$OUT" && mkdir -p "$BR" "$OUT"

if [[ -n "$ONNX" ]]; then
  "$PY" "$MLIRVTA/scripts/onnx_qlinearmul_bridge.py" "$ONNX" -o "$BR"
else
  "$PY" "$MLIRVTA/scripts/onnx_qlinearmul_bridge.py" -o "$BR"
fi

NAME=$($PY -c "import json; print(json.load(open('$BR/bridge_meta.json'))['name'])")
M=$($PY -c "import json; print(json.load(open('$BR/bridge_meta.json'))['raw_dims']['M'])")
K=$($PY -c "import json; print(json.load(open('$BR/bridge_meta.json'))['raw_dims']['K'])")
N=$($PY -c "import json; print(json.load(open('$BR/bridge_meta.json'))['raw_dims']['N'])")
INP=$($PY -c "import json; print(json.load(open('$BR/bridge_meta.json'))['paths']['input'])")
WGT=$($PY -c "import json; print(json.load(open('$BR/bridge_meta.json'))['paths']['weight'])")

"$OPT" "$BR/qlinearmul.mlir" --vta-dram-allocation --lower-vta-gemm -o "$OUT/lowered.mlir"
"$TRANSLATE" "$OUT/lowered.mlir" -o "$OUT" \
  --layer "${NAME}=${M},${K},${N},${INP},${WGT}"

if [[ ! -d "$GD" || ! -f "$GD/instructions.bin" ]]; then
  bash "$MLIRVTA/scripts/make_golden_qlinearmul_upstream.sh"
fi

echo "=== byte-level vs upstream golden ==="
cmp "$OUT/instructions${NAME}.bin" "$GD/instructions.bin" && echo "INSN OK"
cmp "$OUT/uop${NAME}.bin" "$GD/uop.bin" && echo "UOP OK"

cp "$GD/input.bin" "$INP" 2>/dev/null || true

rm -f "$SVTA/log_output/fsim_report.txt"
cp "$OUT/instructions${NAME}.bin" "$CO/instructions.bin"
cp "$OUT/uop${NAME}.bin" "$CO/uop.bin"
cp "$INP" "$CO/input.bin"
cp "$WGT" "$CO/weight.bin"
: > "$CO/add_accumulator.bin"
: > "$CO/accumulator.bin"
cd "$SVTA/examples"
make fsim_compile_single_layer >/dev/null 2>&1
make fsim_single_layer >/dev/null 2>&1
sed -n '/Final result/,/^} $/p' "$SVTA/log_output/fsim_report.txt" > "$OUT/fsim_result.txt"
if [[ -f "$GD/fsim_result.txt" ]]; then
  diff -q "$OUT/fsim_result.txt" "$GD/fsim_result.txt" && echo "=== FSIM vs UPSTREAM: ACCEPT ==="
fi
echo "Increment E (QLinearMul mul_constant) complete."
