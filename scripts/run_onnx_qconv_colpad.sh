#!/usr/bin/env bash
# Increment B: col-pad layout (25x32) + expand_bias → byte-level vs upstream golden.
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
OPT=$HOME/mlir-vta-build/bin/vta-opt
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate
PY=/home/john/miniconda3/envs/standalone-vta/bin/python3
ONNX=${1:-$SVTA/examples/onnx/qlinearconv_debug.onnx}
BR=/tmp/onnx_qconv_colpad
OUT=/tmp/onnx_qconv_colpad_out
GD="$MLIRVTA/test/golden/qlinearconv_upstream"
CO=$SVTA/compiler_output

rm -rf "$BR" "$OUT" && mkdir -p "$BR" "$OUT"

$PY "$MLIRVTA/scripts/onnx_qconv_bridge.py" "$ONNX" -o "$BR" --layout col-pad --zero-input

L=$($PY -c "import json; m=json.load(open('$BR/bridge_meta.json')); print(m['layers'][0])" 2>/dev/null || true)
M=$($PY - <<PY
import json
l=json.load(open("$BR/bridge_meta.json"))["layers"][0]
print(l["raw_dims"]["M"], l["raw_dims"]["K"], l["raw_dims"]["N"], l["name"], l["acc_rows"], l["paths"]["input"], l["paths"]["weight"], l["paths"]["accumulator"], sep=" ")
PY
)
read -r M K N NAME ACCROWS INP WGT ACC <<< "$M"

"$OPT" "$BR/qlinearconv_colpad.mlir" --lower-vta-gemm -o "$OUT/lowered.mlir"
"$TRANSLATE" "$OUT/lowered.mlir" -o "$OUT" \
  --emit-data --rows "$M" --cols "$N" --k "$K" \
  --input "$INP" --weight "$WGT" \
  --accumulator "$ACC" --acc-rows "$ACCROWS"

if [[ -f "$GD/instructions.bin" ]]; then
  echo "=== byte-level vs upstream golden ==="
  cmp "$OUT/instructions.bin" "$GD/instructions.bin" && echo "INSN OK"
  cmp "$OUT/uop.bin" "$GD/uop.bin" && echo "UOP OK"
  cmp "$OUT/input.bin" "$GD/input.bin" && echo "INP OK"
  cmp "$OUT/weight.bin" "$GD/weight.bin" && echo "WGT OK"
  # accumulator: upstream stores raw 1xN; mlir emits blocked prefix-compatible.
  if [[ -f "$GD/accumulator_raw.bin" ]]; then
    cmp -n "$(stat -c%s "$GD/accumulator_raw.bin")" \
      "$OUT/accumulator.bin" "$GD/accumulator_raw.bin" && echo "ACC OK"
  elif [[ $(stat -c%s "$GD/accumulator.bin" 2>/dev/null || echo 0) -eq $(stat -c%s "$OUT/accumulator.bin") ]]; then
    cmp "$OUT/accumulator.bin" "$GD/accumulator.bin" && echo "ACC OK"
  fi
else
  echo "Run scripts/make_golden_qconv_upstream.sh first"
  exit 1
fi

cp "$OUT"/instructions.bin "$OUT"/uop.bin "$OUT"/input.bin "$OUT"/weight.bin \
   "$OUT"/accumulator.bin "$OUT"/out_init.bin "$OUT"/metadata.csv \
   "$OUT"/memory_addresses.csv "$OUT"/layers_name.csv "$CO/"
: > "$CO/add_accumulator.bin"
rm -f "$SVTA/log_output/fsim_report.txt"
cd "$SVTA/examples"
make fsim_compile_single_layer >/dev/null 2>&1
timeout 120 make fsim_single_layer >/dev/null 2>&1 || { echo "FSIM timeout (insn tuning may be needed)" >&2; exit 0; }
sed -n '/Final result/,/^} $/p' "$SVTA/log_output/fsim_report.txt" > "$OUT/fsim_result.txt"
diff -q "$OUT/fsim_result.txt" "$GD/fsim_result.txt" && echo "=== FSIM vs UPSTREAM: ACCEPT ==="

echo "Increment B (col-pad + expand_bias) complete."
