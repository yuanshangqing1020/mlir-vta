#!/usr/bin/env bash
# Conv+Relu two-layer → vta-dram-allocation → gemm+alu → fsim_nn + upstream golden compare.
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
OPT=$HOME/mlir-vta-build/bin/vta-opt
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate
PY=/home/john/miniconda3/envs/standalone-vta/bin/python3
FSIM_DIR=$SVTA/src/simulators/functional_simulator
ONNX=$SVTA/examples/onnx/conv_relu_debug.onnx
GD="$MLIRVTA/test/golden/conv_relu_debug"
BR=/tmp/onnx_conv_relu
OUT=/tmp/onnx_conv_relu_out
CO=$SVTA/compiler_output

if [[ ! -f "$ONNX" ]]; then
  "$PY" "$MLIRVTA/scripts/gen_conv_relu_debug.py"
fi

if [[ ! -f "$GD/reference.bin" ]]; then
  bash "$MLIRVTA/scripts/make_golden_conv_relu_upstream.sh"
fi

rm -rf "$BR" "$OUT" && mkdir -p "$BR" "$OUT"

"$PY" "$MLIRVTA/scripts/onnx_conv_relu_bridge.py" "$ONNX" -o "$BR" --layout col-pad

"$OPT" "$BR/conv_relu.mlir" \
  --vta-dram-allocation --lower-vta-gemm --lower-vta-alu \
  -o "$OUT/lowered.mlir"

LAYER_ARGS=$($PY - <<PY
import json
layers = json.load(open("$BR/bridge_meta.json"))["layers"]
conv = layers[0]
d = conv["raw_dims"]
p = conv["paths"]
print(f'--layer {conv["name"]}={d["M"]},{d["K"]},{d["N"]},{p["input"]},{p["weight"]}')
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
    name = l["name"]
    if "acc_rows" in l and "raw_dims" in l:
        m, n = l["acc_rows"], l["raw_dims"]["N"]
        accp = Path(l["paths"]["accumulator"])
    elif "M" in l:
        m, n = l["M"], 16
        accp = Path(l["paths"]["accumulator_raw"])
    else:
        continue
    acc = np.fromfile(accp, dtype=np.int32).reshape(m, n)
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
import json
from pathlib import Path
layers = json.load(open("$BR/bridge_meta.json"))["layers"]
out = Path("$OUT")
for l in layers:
    if "alu_mlir" not in l:
        continue
    name = l["name"]
    m, n = l["M"], 16
    meta = (
        "Matrix (or Block Size),Nb rows,Nb columns,Is it square?\\r\\n"
        "BS,16,16,True\\r\\n"
        "A,0,0,True\\r\\n"
        f"X,{m},{n},False\\r\\n"
        "Y,0,0,True\\r\\n"
        f"C,{m},{n},True\\r\\n"
    )
    (out / f"metadata{name}.csv").write_text(meta)
    (out / f"weight{name}.bin").write_bytes(b"")
    print(f"Wrote alu metadata/weight for {name}")
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

for l in $($PY -c "import json; print(' '.join(x['name'] for x in json.load(open('$BR/bridge_meta.json'))['layers']))"); do
  : > "$CO/add_accumulator${l}.bin"
done

# Upstream golden input (seed=42) for reproducible numeric compare.
cp "$GD/input_nn.bin" "$CO/input_nn.bin"

for f in "$OUT"/instructions*.bin "$OUT"/uop*.bin "$OUT"/input*.bin "$OUT"/weight*.bin "$OUT"/accumulator*.bin "$OUT"/metadata*.csv "$OUT"/memory_addresses*.csv; do
  [[ -f "$f" ]] && cp "$f" "$CO/"
done

echo "=== Run fsim_nn (Conv+Relu) ==="
cd "$FSIM_DIR"
./build/fsim_nn 2>&1 | grep -E "steps|VTA|written|ERROR" || true

if [[ ! -f "$CO/final_output.bin" ]]; then
  echo "ERROR: final_output.bin not found" >&2
  exit 1
fi

echo "=== Numeric compare vs NumPy QLinearConv+ReLU reference (seed=42) ==="
cmp "$CO/final_output.bin" "$GD/reference.bin"
echo "=== FSIM vs REFERENCE: ACCEPT ==="
echo "Conv+Relu fsim_nn complete."
