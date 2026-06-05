#!/usr/bin/env python3
"""Bridge qconcat_two_branch.onnx → 2× vta.gemm + CPU concat metadata."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnx

MLIRVTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MLIRVTA / "scripts"))

from onnx_qconv_bridge import build_model as build_qconv  # noqa: E402


def _parse_concat_node(model: onnx.ModelProto):
    node = next(n for n in model.graph.node if n.op_type == "QLinearConcat")
    from onnx import numpy_helper

    def _scalar(name: str):
        init = next(t for t in model.graph.initializer if t.name == name)
        arr = numpy_helper.to_array(init)
        return arr.item() if arr.ndim == 0 else arr

    return node, int(_scalar(node.input[1])), float(_scalar(node.input[0]))


def concat_layer_from_convs(
    c1: Dict[str, Any],
    c2: Dict[str, Any],
    *,
    name: str,
    offset_c: int,
    scale_c: float,
) -> Dict[str, Any]:
    _, oc1, oh, ow = c1["output_shape"]
    _, oc2, _, _ = c2["output_shape"]
    assert (oh, ow) == (c2["output_shape"][2], c2["output_shape"][3])
    return {
        "name": name,
        "processor": "concat",
        "reshape": False,
        "offsetA": c1["offsetC"],
        "scaleA": c1["scaleC"],
        "offsetB": c2["offsetC"],
        "scaleB": c2["scaleC"],
        "offsetC": offset_c,
        "scaleC": scale_c,
        "rescaling": 1.0,
        "input_shape": c1["output_shape"],
        "kernel": (1, 1),
        "stride": (1, 1),
        "padding": (0, 0, 0, 0),
        "output_shape": (1, oc1 + oc2, oh, ow),
        "parent_layers": [c1["name"], c2["name"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("/tmp/onnx_qconcat"))
    ap.add_argument("--layout", default="col-pad", choices=["col-pad", "square-pad"])
    args = ap.parse_args()

    qconv = build_qconv(args.onnx, args.out, layout=args.layout, multi=True)
    convs = qconv["layers"]
    if len(convs) != 2:
        raise SystemExit(f"expected 2 QLinearConv layers, got {len(convs)}")
    # Parallel branches: both convs read from image, not chained.
    for c in convs:
        c["parent_layers"] = ["image"]

    model = onnx.load(str(args.onnx))
    concat_node, out_zp, out_scale = _parse_concat_node(model)
    concat = concat_layer_from_convs(
        convs[0],
        convs[1],
        name=concat_node.name or "QLinearConcat1",
        offset_c=out_zp,
        scale_c=out_scale,
    )
    layers = convs + [concat]

    mlir_path = args.out / "qconcat_two_branch.mlir"
    mlir_path.write_text(
        f"func @main({', '.join(c['func_args'] for c in convs)}) {{\n"
        + "".join(c["gemm_mlir"] for c in convs)
        + "  return\n}\n",
        encoding="utf-8",
    )

    meta = {"layers": layers, "mlir": str(mlir_path)}
    meta_json = {
        "mlir": meta["mlir"],
        "layers": [
            {k: v for k, v in l.items() if k not in ("gemm_mlir", "func_args")}
            for l in layers
        ],
    }
    (args.out / "bridge_meta.json").write_text(json.dumps(meta_json, indent=2), encoding="utf-8")
    print(f"MLIR: {mlir_path}  layers={[l['name'] for l in layers]}")


if __name__ == "__main__":
    main()
