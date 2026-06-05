#!/usr/bin/env bash
# Increment C: two-layer QLinearConv (small 16x16) → MLIR multi-layer → fsim_nn.
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
OPT=$HOME/mlir-vta-build/bin/vta-opt
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate
PY=/home/john/miniconda3/envs/standalone-vta/bin/python3
FSIM_DIR=$SVTA/src/simulators/functional_simulator
ONNX=$SVTA/examples/onnx/two_qlinearconv_small.onnx
BR=/tmp/onnx_two_qconv
OUT=/tmp/onnx_two_qconv_out
CO=$SVTA/compiler_output

# Generate small ONNX if missing
if [[ ! -f "$ONNX" ]]; then
  python3 "$MLIRVTA/scripts/gen_two_qlinearconv_small.py"
fi

rm -rf "$BR" "$OUT" && mkdir -p "$BR" "$OUT"

python3 "$MLIRVTA/scripts/onnx_qconv_bridge.py" "$ONNX" -o "$BR" --layout col-pad --multi

"$OPT" "$BR/qlinearconv_multi.mlir" --vta-dram-allocation --lower-vta-gemm -o "$OUT/lowered.mlir"

# Per-layer translate args from bridge_meta.json
LAYER_ARGS=$($PY - <<PY
import json
layers = json.load(open("$BR/bridge_meta.json"))["layers"]
parts = []
for l in layers:
    d = l["raw_dims"]
    p = l["paths"]
    parts.append(f'{l["name"]}={d["M"]},{d["K"]},{d["N"]},{p["input"]},{p["weight"]}')
print(" ".join(f'--layer {x}' for x in parts))
PY
)

# shellcheck disable=SC2086
"$TRANSLATE" "$OUT/lowered.mlir" -o "$OUT" $LAYER_ARGS

# Accumulator per layer (expand_bias 1xN)
$PY - <<PY
import json, numpy as np, struct
from pathlib import Path
layers = json.load(open("$BR/bridge_meta.json"))["layers"]
out = Path("$OUT")
block = 16
for l in layers:
    name = l["name"]
    accp = Path(l["paths"]["accumulator"])
    acc = np.fromfile(accp, dtype=np.int32)
    m, n = l["acc_rows"], l["raw_dims"]["N"]
    acc = acc.reshape(m, n)
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

# dependency.csv for 2 layers
$PY - <<PY
import json, sys
sys.path.insert(0, "$MLIRVTA/scripts")
from emit_dependency_csv import write_dependency_csv
layers = json.load(open("$BR/bridge_meta.json"))["layers"]
# image rows = Ah of first layer
ah = layers[0]["raw_dims"]["ah"]
oc, oh, ow = layers[-1]["output_shape"][1:]
write_dependency_csv(
    "$OUT/dependency.csv", layers,
    image_rows=ah, image_cols=layers[0]["input_shape"][1],
    output_layer=layers[-1]["name"], output_nchw=(oc, oh, ow))
print("dependency.csv written")
PY

cp "$OUT"/dependency.csv "$CO/dependency.csv"
cp "$OUT"/layers_name.csv "$CO/layers_name.csv" 2>/dev/null || true

# input_nn.bin from first layer int8 input
cp "$BR/input_nchw_$(python3 -c "import json;print(json.load(open('$BR/bridge_meta.json'))['layers'][0]['name'])").bin" "$CO/input_nn.bin"

# Copy per-layer artifacts for fsim_nn
for f in "$OUT"/instructions*.bin "$OUT"/uop*.bin "$OUT"/input*.bin "$OUT"/weight*.bin "$OUT"/accumulator*.bin "$OUT"/metadata*.csv "$OUT"/memory_addresses*.csv; do
  [[ -f "$f" ]] && cp "$f" "$CO/"
done

echo "=== Run fsim_nn (2-layer QLinearConv small) ==="
cd "$FSIM_DIR"
./build/fsim_nn 2>&1 | grep -E "steps|VTA|written|ERROR" || true

if [[ -f "$CO/final_output.bin" ]]; then
  echo "=== fsim_nn produced final_output.bin ==="
  wc -c "$CO/final_output.bin"
  echo "Increment C pipeline complete (fsim_nn executed)."
else
  echo "WARNING: final_output.bin not found — check fsim_nn logs" >&2
  exit 1
fi
