#!/usr/bin/env python3
"""Bridge lenet5_mini.onnx → 2× (vta.gemm + vta.alu) for fsim_nn (increment G)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx

MLIRVTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MLIRVTA / "scripts"))

from onnx_qconv_bridge import build_model as build_qconv  # noqa: E402
from onnx_relu_bridge import pad_matrix_to_blocked, to_blocked_bin  # noqa: E402


def relu_layer_from_conv_named(conv: Dict[str, Any], relu_name: str) -> Dict[str, Any]:
    m = conv["raw_dims"]["M"]
    n = conv["raw_dims"]["N"]
    acc = np.zeros((m, n), dtype=np.int32)
    acc_pad, rows = pad_matrix_to_blocked(acc)
    return {
        "name": relu_name,
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
        "func_args": f"%acc_{relu_name}: memref<{rows}x{BLOCK}xi32>",
        "alu_mlir": (
            f"  vta.alu ins(%acc_{relu_name} : memref<{rows}x{BLOCK}xi32>) "
            f'{{name = "{relu_name}", alu_opcode = 1 : i64, use_imm = true, imm = 0 : i64}}\n'
        ),
    }


BLOCK = 16


def _conv_relu_pairs(model: onnx.ModelProto) -> List[Tuple[str, str]]:
    relu_by_input = {n.input[0]: n for n in model.graph.node if n.op_type == "Relu"}
    pairs: List[Tuple[str, str]] = []
    for n in model.graph.node:
        if n.op_type != "QLinearConv":
            continue
        out = n.output[0]
        if out not in relu_by_input:
            raise SystemExit(f"QLinearConv {n.name} not followed by Relu")
        pairs.append((n.name or n.output[0], relu_by_input[out].name or relu_by_input[out].output[0]))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("/tmp/onnx_lenet5_mini"))
    ap.add_argument("--layout", default="col-pad", choices=["col-pad", "square-pad"])
    args = ap.parse_args()

    model = onnx.load(str(args.onnx))
    pairs = _conv_relu_pairs(model)

    qconv = build_qconv(args.onnx, args.out, layout=args.layout, multi=True)
    conv_layers = qconv["layers"]
    if len(conv_layers) != len(pairs):
        raise SystemExit(f"conv count mismatch: {len(conv_layers)} vs {len(pairs)}")

    layers: List[Dict[str, Any]] = []
    gemm_lines: List[str] = []
    alu_lines: List[str] = []
    func_args: List[str] = []

    for idx, (conv_meta, (conv_name, relu_name)) in enumerate(zip(conv_layers, pairs)):
        if idx == 0:
            conv_meta["parent_layers"] = ["image"]
        else:
            conv_meta["parent_layers"] = [layers[-1]["name"]]

        relu = relu_layer_from_conv_named(conv_meta, relu_name)

        acc_path = args.out / f"accumulator_{relu['name']}.bin"
        relu["accumulator"].astype(np.int32).tofile(acc_path)
        blocked = args.out / f"accumulator_blocked_{relu['name']}.bin"
        blocked.write_bytes(to_blocked_bin(relu["accumulator"]))
        relu["paths"] = {
            "accumulator_raw": str(acc_path),
            "accumulator_blocked": str(blocked),
        }
        del relu["accumulator"]

        layers.extend([conv_meta, relu])
        gemm_lines.append(conv_meta["gemm_mlir"])
        alu_lines.append(relu["alu_mlir"])
        func_args.extend([conv_meta["func_args"], relu["func_args"]])

    mlir_path = args.out / "lenet5_mini.mlir"
    mlir_path.write_text(
        f"func @main({', '.join(func_args)}) {{\n"
        + "".join(gemm_lines)
        + "".join(alu_lines)
        + "  return\n}\n",
        encoding="utf-8",
    )

    meta_json = {
        "mlir": str(mlir_path),
        "layers": [
            {k: v for k, v in l.items() if k not in ("gemm_mlir", "func_args", "alu_mlir", "accumulator")}
            for l in layers
        ],
    }
    (args.out / "bridge_meta.json").write_text(json.dumps(meta_json, indent=2), encoding="utf-8")
    print(f"MLIR: {mlir_path}  layers={[l['name'] for l in layers]}")


if __name__ == "__main__":
    main()
