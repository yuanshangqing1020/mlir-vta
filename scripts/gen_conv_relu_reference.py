#!/usr/bin/env python3
"""Generate input_nn.bin + reference.bin for conv_relu_debug (seed=42).

Uses NumPy QLinearConv + int8 ReLU (ORT-compatible semantics). Upstream
reference_onnx cannot run on this ONNX (Relu on int8); nn_compiler also
fails on small Relu shapes — this script is the mlir-vta golden source.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "standalone-vta" / "src" / "compiler"))

from reference_computation.reference_onnx import flatten_conv_output  # noqa: E402
from utils.numpy_implementation import NumPyReferenceEngine  # noqa: E402
from utils.read_csv import load_csv_to_dict  # noqa: E402


def generate_reference(onnx_path: Path, dep_path: Path, out_dir: Path) -> None:
    dep = load_csv_to_dict(str(dep_path))
    first_layer = dep["0"][2]
    attrs = dep[first_layer]
    shape = [1, int(attrs[11]), int(attrs[12]), int(attrs[13])]

    np.random.seed(42)
    input_data = np.random.randint(-128, 127, size=shape).astype(np.int8)

    engine = NumPyReferenceEngine(str(onnx_path))
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
    output_data.tofile(out_dir / "reference.bin")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=Path)
    ap.add_argument("dependency_csv", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    generate_reference(args.onnx, args.dependency_csv, args.out)
    print(f"Wrote {args.out}/input_nn.bin and reference.bin")


if __name__ == "__main__":
    main()
