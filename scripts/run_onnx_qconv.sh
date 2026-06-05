#!/usr/bin/env bash
# run_onnx_qconv.sh — Phase 4 increment A: qlinearconv_debug.onnx → MLIR → binary → FSIM
#
# 1. Python ONNX bridge (im2col + pad → raw bins + vta.gemm MLIR)
# 2. vta-opt --lower-vta-gemm (strategy 2)
# 3. vta-translate --emit-data
# 4. emit dependency.csv
# 5. Byte-level cmp vs golden (if present) + FSIM numerical check
set -euo pipefail

ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
OPT=$HOME/mlir-vta-build/bin/vta-opt
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate
ONNX=${1:-$SVTA/examples/onnx/qlinearconv_debug.onnx}
BR=/tmp/onnx_qconv_run
OUT=/tmp/onnx_qconv_out
GD="$MLIRVTA/test/golden/qlinearconv_padded"
CO=$SVTA/compiler_output

rm -rf "$BR" "$OUT" && mkdir -p "$BR" "$OUT"

echo "=== Step 1: ONNX bridge ==="
python3 "$MLIRVTA/scripts/onnx_qconv_bridge.py" "$ONNX" -o "$BR"

M=$(python3 -c "import json; d=json.load(open('$BR/bridge_meta.json')); print(d['raw_dims']['M'])")
K=$(python3 -c "import json; d=json.load(open('$BR/bridge_meta.json')); print(d['raw_dims']['K'])")
N=$(python3 -c "import json; d=json.load(open('$BR/bridge_meta.json')); print(d['raw_dims']['N'])")
AH=$(python3 -c "import json; d=json.load(open('$BR/bridge_meta.json')); print(d['raw_dims']['ah'])")
NAME=$(python3 -c "import json; d=json.load(open('$BR/bridge_meta.json')); print(d['name'])")

echo "=== Step 2: lower-vta-gemm (strategy 2, ${M}x${K}x${N}) ==="
"$OPT" "$BR/qlinearconv_padded.mlir" --lower-vta-gemm -o "$OUT/lowered.mlir"

echo "=== Step 3: vta-translate ==="
"$TRANSLATE" "$OUT/lowered.mlir" -o "$OUT" \
  --emit-data --rows "$M" --cols "$N" --k "$K" \
  --input "$BR/input_${M}x${K}.bin" \
  --weight "$BR/weight_${K}x${N}.bin"

# Overwrite zero accumulator with bias-expanded padded matrix (blocked layout matches upstream)
python3 - "$BR/accumulator_${M}x${N}.bin" "$OUT/accumulator.bin" "$M" "$N" <<'PY'
import numpy as np, sys
src, dst, m, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
acc = np.fromfile(src, dtype=np.int32).reshape(m, n)
block = 16
mb, nb = m // block, n // block
out = []
for bi in range(mb):
    for bj in range(nb):
        for r in range(block):
            for c in range(block):
                out.append(acc[bi * block + r, bj * block + c])
np.array(out, dtype=np.int32).tofile(dst)
print(f"Wrote blocked accumulator.bin ({len(out)} elems)")
PY

echo "=== Step 4: dependency.csv ==="
python3 "$MLIRVTA/scripts/emit_dependency_csv.py" -o "$OUT/dependency.csv" \
  --layer-name "$NAME" --image-rows "$AH" --image-cols 3

if [[ -d "$GD" && -f "$GD/instructions.bin" ]]; then
  echo "=== Step 5a: byte-level cmp vs golden ==="
  cmp "$OUT/instructions.bin" "$GD/instructions.bin" && echo "INSN OK"
  cmp "$OUT/uop.bin" "$GD/uop.bin" && echo "UOP OK"
else
  echo "=== Step 5a: skip byte cmp (run scripts/make_golden_onnx_qconv.sh first) ==="
fi

echo "=== Step 5b: FSIM single-layer ==="
cp "$OUT"/instructions.bin "$OUT"/uop.bin "$OUT"/input.bin "$OUT"/weight.bin \
   "$OUT"/accumulator.bin "$OUT"/out_init.bin "$OUT"/metadata.csv \
   "$OUT"/memory_addresses.csv "$OUT"/layers_name.csv "$CO/"
: > "$CO/add_accumulator.bin"
rm -f "$SVTA/log_output/fsim_report.txt"
cd "$SVTA/examples"
make fsim_compile_single_layer >/dev/null 2>&1
make fsim_single_layer >/dev/null 2>&1
sed -n '/Final result/,/^} $/p' "$SVTA/log_output/fsim_report.txt" > "$OUT/fsim_result.txt"

echo "=== Step 6: FSIM numerical vs golden ==="
if [[ -f "$GD/fsim_result.txt" ]]; then
  if diff -q "$OUT/fsim_result.txt" "$GD/fsim_result.txt" >/dev/null; then
    echo "=== FSIM vs GOLDEN: ACCEPT ==="
  else
    echo "=== FSIM vs GOLDEN: REJECT ===" >&2
    diff "$GD/fsim_result.txt" "$OUT/fsim_result.txt" >&2 | head -20 || true
    exit 1
  fi
else
  echo "=== FSIM vs GOLDEN: skip (no golden) ==="
fi

echo "dependency.csv at $OUT/dependency.csv"
echo "Phase 4 increment A pipeline complete."
