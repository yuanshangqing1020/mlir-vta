#!/usr/bin/env python3
"""Reference bins for lenet5_mini (2× QLinearConv + Relu)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "standalone-vta" / "src" / "compiler"))

from reference_computation.reference_onnx import flatten_conv_output  # noqa: E402
from utils.numpy_implementation import NumPyReferenceEngine  # noqa: E402


def generate_reference(onnx_path: Path, out_dir: Path, seed: int = 42) -> None:
    np.random.seed(seed)
    engine = NumPyReferenceEngine(str(onnx_path))
    shape = [d.dim_value for d in engine.graph.input[0].type.tensor_type.shape.dim]
    input_data = np.random.randint(-128, 127, size=shape).astype(np.int8)
    input_name = engine.graph.input[0].name
    engine.tensors[input_name] = input_data

    for node in engine.graph.node:
        if node.op_type == "QLinearConv":
            inputs = [engine._get_val(i) for i in node.input]
            engine.tensors[node.output[0]] = engine._qlinear_conv(node, inputs)
        elif node.op_type == "Relu":
            x = engine.tensors[node.input[0]]
            engine.tensors[node.output[0]] = np.maximum(x, 0).astype(np.int8)

    output_data = engine.tensors[engine.graph.output[0].name]
    matrix = flatten_conv_output(input_data.astype(np.int8))

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix.astype(np.int8).tofile(out_dir / "input_nn.bin")
    output_data.astype(np.int8).tofile(out_dir / "reference.bin")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate_reference(args.onnx, args.out, seed=args.seed)
    print(f"Wrote {args.out}/input_nn.bin and reference.bin")


if __name__ == "__main__":
    main()
