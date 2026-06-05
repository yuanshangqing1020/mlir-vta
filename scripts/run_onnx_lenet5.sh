#!/usr/bin/env bash
# Increment G: LeNet-5 mini (2× Conv+Relu) → fsim_nn + numeric golden.
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
OPT=$HOME/mlir-vta-build/bin/vta-opt
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate
PY=/home/john/miniconda3/envs/standalone-vta/bin/python3
FSIM_DIR=$SVTA/src/simulators/functional_simulator
ONNX=$SVTA/examples/onnx/lenet5_mini.onnx
GD="$MLIRVTA/test/golden/lenet5_mini"
BR=/tmp/onnx_lenet5_mini
OUT=/tmp/onnx_lenet5_mini_out
CO=$SVTA/compiler_output

if [[ ! -f "$ONNX" ]] || [[ "$(/home/john/miniconda3/envs/standalone-vta/bin/python3 -c "import onnx; m=onnx.load('$ONNX'); print(list(m.graph.input[0].type.tensor_type.shape.dim[2].dim_value))" 2>/dev/null)" != "5" ]]; then
  "$PY" "$MLIRVTA/scripts/gen_lenet5_mini.py"
fi

rm -rf "$BR" "$OUT" && mkdir -p "$BR" "$OUT" "$GD"

"$PY" "$MLIRVTA/scripts/onnx_lenet5_mini_bridge.py" "$ONNX" -o "$BR" --layout col-pad
"$PY" "$MLIRVTA/scripts/gen_lenet5_mini_reference.py" "$ONNX" -o "$GD"

"$OPT" "$BR/lenet5_mini.mlir" \
  --vta-dram-allocation --lower-vta-gemm --lower-vta-alu \
  -o "$OUT/lowered.mlir"

LAYER_ARGS=$($PY - <<PY
import json
layers = json.load(open("$BR/bridge_meta.json"))["layers"]
parts = []
for l in layers:
    if l.get("processor") != "vta" or "raw_dims" not in l:
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
    if l.get("processor") != "vta" or "M" not in l:
        continue
    if "raw_dims" in l:
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
ah = conv["raw_dims"]["ah"]
oc, oh, ow = layers[-1]["output_shape"][1:]
write_dependency_csv(
    "$OUT/dependency.csv", layers,
    image_rows=ah,
    image_cols=conv["input_shape"][1],
    output_layer=layers[-1]["name"],
    output_nchw=(oc, oh, ow))
print("dependency.csv written")
PY

cp "$OUT/dependency.csv" "$CO/dependency.csv"
cp "$OUT/layers_name.csv" "$CO/layers_name.csv" 2>/dev/null || true
cp "$GD/input_nn.bin" "$CO/input_nn.bin"

for l in $($PY -c "import json; print(' '.join(x['name'] for x in json.load(open('$BR/bridge_meta.json'))['layers']))"); do
  : > "$CO/add_accumulator${l}.bin"
done

for f in "$OUT"/instructions*.bin "$OUT"/uop*.bin "$OUT"/input*.bin "$OUT"/weight*.bin \
         "$OUT"/accumulator*.bin "$OUT"/metadata*.csv "$OUT"/memory_addresses*.csv; do
  [[ -f "$f" ]] && cp "$f" "$CO/"
done

echo "=== Run fsim_nn (LeNet-5 mini: 2× Conv+Relu) ==="
cd "$FSIM_DIR"
./build/fsim_nn 2>&1 | grep -E "steps|VTA|written|ERROR|Relu" || true

[[ -f "$CO/final_output.bin" ]] || { echo "ERROR: final_output.bin missing" >&2; exit 1; }

echo "=== Numeric compare vs NumPy reference (seed=42) ==="
cmp "$CO/final_output.bin" "$GD/reference.bin"
echo "=== FSIM vs REFERENCE: ACCEPT (G LeNet-5 mini) ==="
