#!/usr/bin/env bash
# Generate a byte-level + FSIM golden for a pure GEMM C[MxN] = A[MxK] * B[KxN]
# using the upstream standalone-vta Python compiler, into
#   mlir-vta/test/golden/matmul_<M>x<K>x<N>/
# Usage: scripts/make_golden_gemm.sh <M> <K> <N>   (all multiples of 16)
set -euo pipefail
M=${1:?M}; K=${2:?K}; N=${3:?N}
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
TAG="${M}x${K}x${N}"
GD="$MLIRVTA/test/golden/matmul_${TAG}"
CO="$SVTA/compiler_output"
mkdir -p "$GD"

# 1. fixed random raw matrices (seed 42): input MxK, weight KxN
python3 - "$M" "$K" "$N" <<'PY'
import numpy as np, sys
M,K,N = map(int, sys.argv[1:4])
np.random.seed(42)
np.random.randint(-8,8,size=(M,K),dtype=np.int32).tofile(f"/mnt/c/MLIR-VTA/standalone-vta/compiler_output/input_{M}x{K}.bin")
np.random.randint(-8,8,size=(K,N),dtype=np.int32).tofile(f"/mnt/c/MLIR-VTA/standalone-vta/compiler_output/weight_{K}x{N}.bin")
print(f"fixed input {M}x{K} / weight {K}x{N} written")
PY

# 2. VTA IR JSON (pure GEMM)
JSON="$SVTA/examples/vta_ir/matmul_${TAG}.json"
cat > "$JSON" <<EOF
{
  "NAME": "",
  "MATRICES" : {
    "A": [${M}, ${K}, "../compiler_output/input_${M}x${K}.bin"],
    "B": [${K}, ${N}, "../compiler_output/weight_${K}x${N}.bin"],
    "C": [${M}, ${N}, "output"]
  },
  "LOAD": { "INP": ["A"], "WGT": ["B"] },
  "GEMM": ["C", "A", "B"],
  "STORE": { "C": ["C"] }
}
EOF
cp "$JSON" "$CO/"

# 3. compile (debug off, summary on)
cd "$SVTA/examples"
python3 ../src/compiler/vta_compiler/main_vta_compiler.py False True False \
    ../config/vta_config.json "../compiler_output/matmul_${TAG}.json" >/dev/null

# 4. run FSIM to capture result matrix
: > "$CO/add_accumulator.bin"
rm -f "$SVTA/log_output/fsim_report.txt"
make fsim_compile_single_layer >/dev/null 2>&1
make fsim_single_layer >/dev/null 2>&1

# 5. collect golden artifacts
cp "$CO"/instructions.bin "$CO"/uop.bin "$CO"/input.bin "$CO"/weight.bin \
   "$CO"/accumulator.bin "$CO"/memory_addresses.csv "$CO"/metadata.csv \
   "$CO"/layers_name.csv \
   "$CO/input_${M}x${K}.bin" "$CO/weight_${K}x${N}.bin" "$JSON" "$GD/"
sed -n '/Final result/,/^} $/p' "$SVTA/log_output/fsim_report.txt" \
    > "$GD/fsim_result_${TAG}.txt"
echo "golden for ${TAG} written to $GD"
ls "$GD"
