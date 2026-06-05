#!/usr/bin/env python3
"""Bridge qadd_two_mul.onnx → 2× vta.gemm(mul_constant) + qadd dependency metadata."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnx
from onnx import numpy_helper

MLIRVTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MLIRVTA / "scripts"))

from onnx_qlinearmul_bridge import build_standalone_16x16, gemm_from_meta  # noqa: E402


def qadd_layer_from_muls(m1: Dict[str, Any], m2: Dict[str, Any], add_name: str) -> Dict[str, Any]:
    shape = m1["output_shape"]
    return {
        "name": add_name,
        "processor": "qadd",
        "reshape": "int32",
        "offsetA": m1["offsetC"],
        "scaleA": m1["scaleC"],
        "offsetB": m2["offsetC"],
        "scaleB": m2["scaleC"],
        "offsetC": m1["offsetC"],
        "scaleC": m1["scaleC"],
        "rescaling": 1.0,
        "input_shape": shape,
        "kernel": (1, 1),
        "stride": (1, 1),
        "padding": (0, 0, 0, 0),
        "output_shape": shape,
        "parent_layers": [m1["name"], m2["name"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("/tmp/onnx_qadd_two_mul"))
    args = ap.parse_args()

    model = onnx.load(str(args.onnx))
    mul_nodes = [n for n in model.graph.node if n.op_type == "QLinearMul"]
    add_node = next(n for n in model.graph.node if n.op_type == "QLinearAdd")
    if len(mul_nodes) != 2:
        raise SystemExit(f"expected 2 QLinearMul, got {len(mul_nodes)}")

    args.out.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    base = build_standalone_16x16(name="_shared", scalar=1)
    base["input_matrix"] = np.random.randint(-8, 8, size=(16, 16), dtype=np.int32)

    layers: List[Dict[str, Any]] = []
    gemm_lines: List[str] = []
    func_args: List[str] = []
    for node in mul_nodes:
        scalar = int(numpy_helper.to_array(
            next(t for t in model.graph.initializer if t.name == node.input[0])
        ).reshape(-1)[0])
        meta = dict(base)
        meta["name"] = node.name
        meta["scalar"] = scalar
        layer = gemm_from_meta(meta, args.out)
        # Shared input bin for both branches.
        if not layers:
            shared_in = Path(layer["paths"]["input"])
        else:
            layer["paths"]["input"] = str(shared_in)
        layers.append(layer)
        gemm_lines.append(layer["gemm_mlir"])
        func_args.append(layer["func_args"])

    qadd = qadd_layer_from_muls(layers[0], layers[1], add_node.name or "QLinearAdd1")
    layers.append(qadd)

    mlir_path = args.out / "qadd_two_mul.mlir"
    mlir_path.write_text(
        f"func @main({', '.join(func_args)}) {{\n"
        + "".join(gemm_lines)
        + "  return\n}\n",
        encoding="utf-8",
    )

    meta = {"layers": layers, "shared_input": str(shared_in), "mlir": str(mlir_path)}
    (args.out / "bridge_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"MLIR: {mlir_path}  layers={[l['name'] for l in layers]}")


if __name__ == "__main__":
    main()
