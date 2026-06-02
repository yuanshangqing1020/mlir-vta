#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
OUT=/tmp/vtaout_fsim_linalg
OPT=$HOME/mlir-vta-build/bin/vta-opt
TRANSLATE=$HOME/mlir-vta-build/bin/vta-translate

rm -rf "$OUT" && mkdir -p "$OUT"
# 1. tensor 入口 → 完整管道 → vtaisa 程序 (PIPELINE 与 _spike_bufferize_notes.md 一致)
"$OPT" "$MLIRVTA/test/Target/matmul_tensor.mlir" \
  --linalg-tile="linalg-tile-sizes=16,16,16" \
  --linalg-bufferize --tensor-bufferize --tensor-constant-bufferize \
  --func-bufferize --finalizing-bufferize \
  --convert-linalg-to-vta --canonicalize --lower-vta-gemm \
  -o "$OUT/lowered.mlir"
# 2. 发射二进制 + 数据/CSV
"$TRANSLATE" "$OUT/lowered.mlir" -o "$OUT" \
  --emit-data --input "$MLIRVTA/test/golden/input_16x16.bin" \
  --weight "$MLIRVTA/test/golden/weight_16x16.bin"
# 3. stage 进 standalone-vta 并跑 FSIM
cp "$OUT"/instructions.bin "$OUT"/uop.bin "$OUT"/input.bin "$OUT"/weight.bin \
   "$OUT"/accumulator.bin "$OUT"/out_init.bin "$OUT"/metadata.csv \
   "$OUT"/memory_addresses.csv "$OUT"/layers_name.csv "$SVTA/compiler_output/"
: > "$SVTA/compiler_output/add_accumulator.bin"
cd "$SVTA/examples"
make fsim_compile_single_layer
make fsim_single_layer
echo "===== FSIM REPORT (tail) ====="
tail -40 "$SVTA/log_output/fsim_report.txt"
