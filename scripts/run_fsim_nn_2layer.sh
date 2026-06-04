#!/usr/bin/env bash
# run_fsim_nn_2layer.sh — 真·层间串联验证：2 层 16×16 GEMM 经 fsim_nn 串联执行
#
# 流程：
#   1. 生成测试数据（seed=42，[-2,2] 小范围整数）
#   2. 用上游 VTA 编译器同时编译 NNA+NNB（连续 DRAM 地址）
#   3. 生成 input_nn.bin（int8）和 dependency.csv / layers_name.csv
#   4. 运行 fsim_nn（2 步串联：NNA → NNB，NNA 输出作 NNB 输入）
#   5. 用 fsim_single_layer（NNB 以 NNA 输出为输入）独立验证
#   6. 对比 fsim_nn final_output.bin 与独立验证结果（256 元素完全一致）
#
# 注意：fsim_nn 需要多层连续 DRAM 地址，必须一次编译两层。
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MLIR_VTA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SVTA=/mnt/c/MLIR-VTA/standalone-vta
CO=$SVTA/compiler_output
VIR=$SVTA/examples/vta_ir
FSIM_DIR=$SVTA/src/simulators/functional_simulator
PYTHON=python3

echo "=== Step 1: Generate test data (seed=42, [-2,2]) ==="
$PYTHON << 'PYEOF'
import numpy as np, json, os

CO = '/mnt/c/MLIR-VTA/standalone-vta/compiler_output'
VIR = '/mnt/c/MLIR-VTA/standalone-vta/examples/vta_ir'

np.random.seed(42)
A  = np.random.randint(-2, 3, (16, 16), dtype=np.int32)
W0 = np.random.randint(-2, 3, (16, 16), dtype=np.int32)
W1 = np.random.randint(-2, 3, (16, 16), dtype=np.int32)

# Save input and weights
A.flatten().tofile(f'{CO}/input_nn_2layer.bin')
W0.flatten().tofile(f'{CO}/weight_nn_L0.bin')
W1.flatten().tofile(f'{CO}/weight_nn_L1.bin')

# input_nn.bin: int8 for fsim_nn
A.astype(np.int8).flatten().tofile(f'{CO}/input_nn.bin')

# VTA IR JSON files
for name, w_file in [('NNA', 'weight_nn_L0.bin'), ('NNB', 'weight_nn_L1.bin')]:
    ir = {
        'NAME': name,
        'MATRICES': {
            'A': [16, 16, f'../compiler_output/input_nn_2layer.bin'],
            'B': [16, 16, f'../compiler_output/{w_file}'],
            'C': [16, 16, 'output']
        },
        'LOAD': {'INP': ['A'], 'WGT': ['B']},
        'GEMM': ['C', 'A', 'B'],
        'STORE': {'C': ['C']}
    }
    with open(f'{VIR}/matmul_{name}.json', 'w') as f:
        json.dump(ir, f, indent=2)

print('Data generated.')
PYEOF

echo ""
echo "=== Step 2: Compile NNA+NNB with continuous DRAM addresses ==="
cd $SVTA/examples
$PYTHON ../src/compiler/vta_compiler/main_vta_compiler.py True True False \
    ../config/vta_config.json vta_ir/matmul_NNA.json vta_ir/matmul_NNB.json \
    2>&1 | grep -E "nb_steps|nb_uop|nb_insn|TOTAL"

echo ""
echo "=== Step 3: Write dependency.csv and layers_name.csv ==="
$PYTHON << 'PYEOF'
CO = '/mnt/c/MLIR-VTA/standalone-vta/compiler_output'

layers_name = """Line identifier,Nb of VTA IR,Provide execution log
nb_vta_ir,2,True
Line identifier,VTA IR name,Last physical DRAM address allocated by the layer
0,NNA,0x417f
1,NNB,0x417f"""

dependency = """Line identifier,Number of layers
nb_steps,2
Line identifier,Nb rows,Nb columns
image,16,16
Line identifier,Final layer name,Output tensor channels,Output tensor height,Output tensor width
output,NNB,16,1,16
Execution order,Processor,Layer name
0,vta,NNA
1,vta,NNB
Layer name,Processor,Reshape,offsetA,scaleA,offsetB,scaleB,offsetU,scaleU,offsetV,scaleV,Input channels,Input height,Input width,Kernel height,Kernel width,Stride height,Stride width,Padding top,Padding left,Padding bottom,Padding right,Output channel,Output height,Output width,offsetC,scaleC,Rescaling factor,Parent layers
NNA,vta,im2row,0,1,0,1,0,1,0,1,16,1,16,1,1,1,1,0,0,0,0,16,1,16,0,1,1,INP,1,image
NNB,vta,im2row,0,1,0,1,0,1,0,1,16,1,16,1,1,1,1,0,0,0,0,16,1,16,0,1,1,INP,1,NNA"""

with open(f'{CO}/layers_name.csv', 'w') as f:
    f.write(layers_name)
with open(f'{CO}/dependency.csv', 'w') as f:
    f.write(dependency)
print('CSVs written.')
PYEOF

echo ""
echo "=== Step 4: Run fsim_nn (2-layer chain) ==="
cd $FSIM_DIR
./build/fsim_nn 2>&1 | grep -E "steps|VTA|written"

echo ""
echo "=== Step 5: Independent validation via fsim_single_layer ==="
# Compile NNB_test with NNA output as input
$PYTHON << 'PYEOF'
import numpy as np, json

CO = '/mnt/c/MLIR-VTA/standalone-vta/compiler_output'
VIR = '/mnt/c/MLIR-VTA/standalone-vta/examples/vta_ir'

# First run fsim_nn to get NNA outC
# We'll use the outC saved during the step 4 run... but it needs fsim_nn with debug
# Alternative: run fsim_single_layer for NNA to get its result, save as inputNNB_test.bin

# Since NNA outC = VTA GEMM(inputNNA.bin, weightNNA.bin), re-run single-layer NNA
# and save outC using fsim_single_layer's internal result
# Actually: we can approximate by using fsim_single_layer for NNA and parsing the output

# More simply: use the inputNNA.bin (which is A) processed through VTA GEMM
# The result of NNA = outC of NNA, which was saved when fsim_nn ran in debug mode
# But we cleaned up the debug code. Let's just verify mathematically:

# The fsim_nn test is already complete (step 4). Now we just need to compare
# final_output.bin with what fsim_single_layer gives for NNB with NNA output as input.

# We'll use the NNA outC from the KNOWN correct run (fsim_single_layer NNA first)
# Setup layers_name.csv for single-layer NNA
layers_nna = """Line identifier,Nb of VTA IR,Provide execution log
nb_vta_ir,1,True
Line identifier,VTA IR name,Last physical DRAM address allocated by the layer
0,NNA,0x417f"""
with open(f'{CO}/layers_name.csv', 'w') as f:
    f.write(layers_nna)
print('Set up for NNA single-layer run.')
PYEOF

cd $FSIM_DIR
# Run NNA single-layer, capture output
./build/fsim_single_layer 2>&1 > /tmp/fsim_nna_out.txt
echo "NNA single-layer first row: $(grep -A4 'block_id: 0' /tmp/fsim_nna_out.txt | head -2 | tail -1)"

# Now save NNA's outC by looking at inputNNA.bin
# outC_NNA = the 256 int32 VTA output of NNA
# We know NNA outC[0..3] = [-12, 8, -9, 7] from fsim_single_layer first row
# Use inputNNA.bin as-is (since it = inputNNA before, and we're setting it from outC)

# Actually use the final_output.bin comparison done via Python to fully validate
$PYTHON << 'PYEOF2'
import numpy as np
import subprocess

CO = '/mnt/c/MLIR-VTA/standalone-vta/compiler_output'
FSIM = '/mnt/c/MLIR-VTA/standalone-vta/src/simulators/functional_simulator'

# Parse NNA result from fsim_single_layer output 
with open('/tmp/fsim_nna_out.txt') as f:
    content = f.read()

# Extract matrix values from "Final result = {" section
lines = content.split('\n')
matrix_lines = []
in_block = False
for line in lines:
    if 'block_id: 0' in line:
        in_block = True
        continue
    if in_block:
        if '}' in line:
            break
        # Remove leading/trailing whitespace and parse numbers
        nums = line.strip().split()
        if nums:
            matrix_lines.append([int(x) for x in nums])

if len(matrix_lines) == 16:
    nna_result = np.array(matrix_lines, dtype=np.int32)
    print('NNA result row 0:', nna_result[0, :8])
    
    # NNA outC (int32) is needed as inputNNB.bin for NNB single-layer test
    # Save as int32 flat
    nna_result.flatten().tofile(f'{CO}/inputNNB_from_NNA.bin')
    print('Saved NNA result as inputNNB_from_NNA.bin')
else:
    print(f'WARNING: Got {len(matrix_lines)} matrix lines, expected 16')
    print('Skipping NNB validation step')
PYEOF2

# Compile NNB with NNA output as input (single-layer from addr 0)
$PYTHON -c "
import json
VIR = '/mnt/c/MLIR-VTA/standalone-vta/examples/vta_ir'
CO = '/mnt/c/MLIR-VTA/standalone-vta/compiler_output'
ir = {
    'NAME': 'NNB_fromNNA',
    'MATRICES': {
        'A': [16, 16, '../compiler_output/inputNNB_from_NNA.bin'],
        'B': [16, 16, '../compiler_output/weight_nn_L1.bin'],
        'C': [16, 16, 'output']
    },
    'LOAD': {'INP': ['A'], 'WGT': ['B']},
    'GEMM': ['C', 'A', 'B'],
    'STORE': {'C': ['C']}
}
with open(f'{VIR}/matmul_NNB_fromNNA.json', 'w') as f:
    json.dump(ir, f, indent=2)
print('Created NNB_fromNNA IR')
"

cd $SVTA/examples
$PYTHON ../src/compiler/vta_compiler/main_vta_compiler.py True True False \
    ../config/vta_config.json vta_ir/matmul_NNB_fromNNA.json 2>&1 | tail -3

$PYTHON -c "
CO = '/mnt/c/MLIR-VTA/standalone-vta/compiler_output'
c = '''Line identifier,Nb of VTA IR,Provide execution log
nb_vta_ir,1,True
Line identifier,VTA IR name,Last physical DRAM address allocated by the layer
0,NNB_fromNNA,0x417f'''
with open(f'{CO}/layers_name.csv', 'w') as f:
    f.write(c)
"
cd $FSIM_DIR
./build/fsim_single_layer 2>&1 > /tmp/fsim_nnb_from_nna_out.txt
echo "NNB_fromNNA first row: $(grep -A4 'block_id: 0' /tmp/fsim_nnb_from_nna_out.txt | head -2 | tail -1)"

echo ""
echo "=== Step 6: Compare fsim_nn final_output.bin vs NNB_fromNNA single-layer ==="
$PYTHON << 'PYEOF3'
import numpy as np

CO = '/mnt/c/MLIR-VTA/standalone-vta/compiler_output'

# Read fsim_nn final output
final_raw = np.fromfile(f'{CO}/final_output.bin', dtype=np.int8)

# Decode NCHW [1,16,1,16] to row-major matrix
final_matrix = np.zeros((16, 16), dtype=np.int8)
for c in range(16):
    for r in range(16):
        final_matrix[r][c] = final_raw[c*16 + r]

# Parse fsim_single_layer NNB_fromNNA result
with open('/tmp/fsim_nnb_from_nna_out.txt') as f:
    content = f.read()

lines = content.split('\n')
matrix_lines = []
in_block = False
for line in lines:
    if 'block_id: 0' in line:
        in_block = True
        continue
    if in_block:
        if '}' in line:
            break
        nums = line.strip().split()
        if nums:
            matrix_lines.append([int(x) for x in nums])

if len(matrix_lines) == 16:
    ref_matrix = np.array(matrix_lines, dtype=np.int8)
    
    if np.array_equal(final_matrix, ref_matrix):
        print()
        print('=== FSIM_NN 2-LAYER GEMM CHAIN: ACCEPT (256/256 elements match) ===')
        print('fsim_nn (NNA→NNB serial) == fsim_single_layer(NNB with NNA output)')
    else:
        diff = np.sum(final_matrix != ref_matrix)
        print(f'MISMATCH: {diff}/256 elements differ')
        for r in range(16):
            if not np.array_equal(final_matrix[r], ref_matrix[r]):
                print(f'  row {r}: fsim_nn={final_matrix[r]}, ref={ref_matrix[r]}')
else:
    print(f'WARNING: Could not parse reference matrix ({len(matrix_lines)} rows)')
PYEOF3

# Restore dependency.csv and layers_name.csv for 2-layer test
$PYTHON << 'PYEOF4'
CO = '/mnt/c/MLIR-VTA/standalone-vta/compiler_output'
layers_name = """Line identifier,Nb of VTA IR,Provide execution log
nb_vta_ir,2,True
Line identifier,VTA IR name,Last physical DRAM address allocated by the layer
0,NNA,0x417f
1,NNB,0x417f"""
with open(f'{CO}/layers_name.csv', 'w') as f:
    f.write(layers_name)
PYEOF4
