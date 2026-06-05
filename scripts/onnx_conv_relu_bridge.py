#!/usr/bin/env python3
"""Bridge QLinearConv -> Relu ONNX to multi-layer MLIR (gemm + alu) for fsim_nn."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

MLIRVTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MLIRVTA / "scripts"))

from onnx_qconv_bridge import build_model as build_qconv  # noqa: E402
from onnx_relu_bridge import pad_matrix_to_blocked, to_blocked_bin  # noqa: E402

BLOCK = 16


def relu_layer_from_conv(conv: Dict[str, Any], onnx_path: Path) -> Dict[str, Any]:
    import onnx

    model = onnx.load(str(onnx_path))
    relu = next(n for n in model.graph.node if n.op_type == "Relu")
    name = relu.name or "Relu1"
    m = conv["raw_dims"]["M"]
    n = conv["raw_dims"]["N"]
    acc = np.zeros((m, n), dtype=np.int32)
    acc_pad, rows = pad_matrix_to_blocked(acc)
    _, oc, oh, ow = conv["output_shape"]
    return {
        "name": name,
        "M": rows,
        "processor": "vta",
        "reshape": "int32",
        "offsetA": 0,
        "scaleA": 1.0,
        "offsetB": 0,
        "scaleB": 1.0,
        "offsetC": 0,
        "scaleC": 1.0,
        "rescaling": 1.0,
        "input_shape": conv["output_shape"],
        "kernel": (1, 1),
        "stride": (1, 1),
        "padding": (0, 0, 0, 0),
        "output_shape": conv["output_shape"],
        "parent_layers": [conv["name"]],
        "alu_opcode": 1,
        "use_imm": True,
        "imm": 0,
        "accumulator": acc_pad,
        "func_args": f"%acc_{name}: memref<{rows}x{BLOCK}xi32>",
        "alu_mlir": (
            f"  vta.alu ins(%acc_{name} : memref<{rows}x{BLOCK}xi32>) "
            f'{{name = "{name}", alu_opcode = 1 : i64, use_imm = true, imm = 0 : i64}}\n'
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Conv+Relu → vta.gemm + vta.alu")
    ap.add_argument("onnx", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("/tmp/onnx_conv_relu"))
    ap.add_argument("--layout", default="col-pad", choices=["col-pad", "square-pad"])
    args = ap.parse_args()

    qconv = build_qconv(args.onnx, args.out, layout=args.layout, multi=False)
    conv = qconv["layers"][0]
    relu = relu_layer_from_conv(conv, args.onnx)

    acc_path = args.out / f"accumulator_{relu['name']}.bin"
    relu["accumulator"].astype(np.int32).tofile(acc_path)
    blocked = args.out / f"accumulator_blocked_{relu['name']}.bin"
    blocked.write_bytes(to_blocked_bin(relu["accumulator"]))
    relu["paths"] = {
        "accumulator_raw": str(acc_path),
        "accumulator_blocked": str(blocked),
    }
    del relu["accumulator"]

    args_str = ", ".join([conv["func_args"], relu["func_args"]])
    body = conv["gemm_mlir"] + relu["alu_mlir"]
    mlir_path = args.out / "conv_relu.mlir"
    mlir_path.write_text(f"func @main({args_str}) {{\n{body}  return\n}}\n")

    meta = {"layers": [conv, relu], "mlir": str(mlir_path)}
    (args.out / "bridge_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"MLIR: {mlir_path}  conv={conv['name']} relu={relu['name']}")


if __name__ == "__main__":
    main()
