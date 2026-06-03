#!/usr/bin/env bash
# Byte-level acceptance for faithful multi-layer compilation: two independent
# 16x16 GEMM layers (L0, L1) in one func -> -vta-dram-allocation (cross-layer
# page-aligned base increment) -> lower-vta-gemm -> per-layer artifacts. Checks
# every per-layer file is byte-identical to the upstream two-layer golden.
#
# NOTE on FSIM: the upstream `fsim_single_layer` allocates+frees DRAM per layer
# and reuses low pages, so it cannot execute the upstream compiler's *cumulative*
# multi-layer addresses (L1 INP=0x7000+). The upstream golden binaries crash
# fsim_single_layer identically; true multi-layer execution uses `fsim_nn`
# (deferred). Byte-equality to the reference compiler is the faithfulness gate;
# per-layer compute correctness is covered by the 16x16 single-layer FSIM (L0's
# instruction stream is byte-identical to the standard 16x16 program).
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
MLIRVTA=$ROOT/mlir-vta
OPT=$HOME/mlir-vta-build/bin/vta-opt
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate
GD=$MLIRVTA/test/golden/matmul_2layer_16x16
OUT=/tmp/vtaout_2layer
rm -rf "$OUT" && mkdir -p "$OUT"

# 1. memref-level two-layer entry -> alloc (multi-layer base) -> lower
"$OPT" "$MLIRVTA/test/Target/matmul_2layer_16x16.mlir" \
  -vta-dram-allocation -lower-vta-gemm -o "$OUT/lowered.mlir"

# 2. per-layer instructions/uop + layers_name/memory CSV (binary emitter) and
#    per-layer data/metadata (data emitter, --layer). Both layers reuse the same
#    raw 16x16 input/weight, matching the golden.
"$TRANSLATE" "$OUT/lowered.mlir" -o "$OUT" \
  --layer L0=16,16,16,"$GD/input_16x16.bin","$GD/weight_16x16.bin" \
  --layer L1=16,16,16,"$GD/input_16x16.bin","$GD/weight_16x16.bin"

# 3. byte-level gate against the upstream two-layer golden (15 files)
ok=1
for f in instructionsL0.bin instructionsL1.bin uopL0.bin uopL1.bin layers_name.csv \
         memory_addressesL0.csv memory_addressesL1.csv metadataL0.csv metadataL1.csv \
         inputL0.bin inputL1.bin weightL0.bin weightL1.bin \
         accumulatorL0.bin accumulatorL1.bin; do
  if cmp -s "$OUT/$f" "$GD/$f"; then echo "  $f OK"; else echo "  $f DIFF"; ok=0; fi
done
if [ $ok -eq 1 ]; then
  echo "ALL 15 PER-LAYER FILES BYTE-LEVEL OK (ACCEPT)"
else
  echo "MULTI-LAYER BYTE MISMATCH (REJECT)" >&2
  exit 1
fi
