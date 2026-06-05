#!/usr/bin/env bash
# Increment F-b: two parallel QLinearConv (VTA) + QLinearConcat (CPU) → fsim_nn.
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
OPT=$HOME/mlir-vta-build/bin/vta-opt
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate
PY=/home/john/miniconda3/envs/standalone-vta/bin/python3
FSIM_DIR=$SVTA/src/simulators/functional_simulator
ONNX=$SVTA/examples/onnx/qconcat_two_branch.onnx
GD="$MLIRVTA/test/golden/qconcat_two_branch"
BR=/tmp/onnx_qconcat
OUT=/tmp/onnx_qconcat_out
CO=$SVTA/compiler_output

if [[ ! -f "$ONNX" ]]; then
  "$PY" "$MLIRVTA/scripts/gen_qconcat_two_branch.py"
fi

rm -rf "$BR" "$OUT" && mkdir -p "$BR" "$OUT" "$GD"

"$PY" "$MLIRVTA/scripts/onnx_qconcat_two_branch_bridge.py" "$ONNX" -o "$BR" --layout col-pad

"$PY" "$MLIRVTA/scripts/gen_qconcat_reference.py" "$ONNX" -o "$GD"

"$OPT" "$BR/qconcat_two_branch.mlir" --vta-dram-allocation --lower-vta-gemm -o "$OUT/lowered.mlir"

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
import json, numpy as np
from pathlib import Path
layers = json.load(open("$BR/bridge_meta.json"))["layers"]
out = Path("$OUT")
block = 16
for l in layers:
    if l.get("processor") != "vta":
        continue
    name = l["name"]
    m, n = l["acc_rows"], l["raw_dims"]["N"]
    acc = np.fromfile(l["paths"]["accumulator"], dtype=np.int32).reshape(m, n)
    accMb = (m + block - 1) // block
    Nb = n // block
    blocked = []
    for bi in range(accMb):
        for bj in range(Nb):
            for r in range(block):
                for c in range(block):
                    gr, gc = bi*block+r, bj*block+c
                    blocked.append(int(acc[gr, gc]) if gr < m and gc < n else 0)
    dst = out / f"accumulator{name}.bin"
    np.array(blocked, dtype=np.int32).tofile(dst)
    print(f"Wrote {dst}")
PY

$PY - <<PY
import json, sys
sys.path.insert(0, "$MLIRVTA/scripts")
from emit_dependency_csv import write_dependency_csv
layers = json.load(open("$BR/bridge_meta.json"))["layers"]
conv = layers[0]
oc, oh, ow = layers[-1]["output_shape"][1:]
write_dependency_csv(
    "$OUT/dependency.csv", layers,
    image_rows=conv["raw_dims"]["ah"],
    image_cols=conv["input_shape"][1],
    output_layer=layers[-1]["name"],
    output_nchw=(oc, oh, ow))
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

echo "=== Run fsim_nn (2× Conv + QLinearConcat) ==="
cd "$FSIM_DIR"
if ! ./build/fsim_nn 2>&1 | tee /tmp/fsim_qconcat.log | grep -E "steps|VTA|written|ERROR|concat|Auto-created" ; then
  tail -20 /tmp/fsim_qconcat.log >&2
fi

[[ -f "$CO/final_output.bin" ]] || { echo "ERROR: final_output.bin missing" >&2; exit 1; }

echo "=== Numeric compare vs NumPy reference (seed=42) ==="
cmp "$CO/final_output.bin" "$GD/reference.bin"
echo "=== FSIM vs REFERENCE: ACCEPT (F-b QLinearConcat) ==="
