#!/usr/bin/env bash
set -euo pipefail
SVTA=/mnt/c/MLIR-VTA/standalone-vta
OUT="$(cd "$(dirname "$0")/../test/golden" && pwd)"
cd "$SVTA"
python3 - <<'PY'
import numpy as np
np.random.seed(42)
for name in ("input","weight"):
    a = np.random.randint(-8, 8, size=(16,16), dtype=np.int32)
    a.tofile(f"compiler_output/{name}_16x16.bin")
print("fixed input/weight written")
PY
cp examples/vta_ir/matmul_16x16.json compiler_output/
cd examples
python3 ../src/compiler/vta_compiler/main_vta_compiler.py False True False \
    ../config/vta_config.json ../compiler_output/matmul_16x16.json >/dev/null
cp ../compiler_output/instructions.bin ../compiler_output/uop.bin \
   ../compiler_output/input_16x16.bin ../compiler_output/weight_16x16.bin "$OUT/"
echo "golden refreshed into $OUT"
