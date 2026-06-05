#!/usr/bin/env python3
"""Bridge QLinearMul / MulConstant GEMM to vta.gemm MLIR + bins.

Default: 16x16 standalone (matmulConst_16x16 golden).
Optional ONNX: qlinearmul_debug.onnx (3x3, scalar multiply).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

BLOCK = 16
SVTA = Path("/mnt/c/MLIR-VTA/standalone-vta")
sys.path.insert(0, str(SVTA / "src" / "compiler"))


def _ceil_block(n: int, block: int = BLOCK) -> int:
    return ((n - 1) // block + 1) * block


def matrix_diagonal(scalar: int, rows: int, cols: int) -> np.ndarray:
    """Upstream-style diagonal WGT for MulConstant."""
    out = np.zeros((rows, cols), dtype=np.int32)
    n = min(rows, cols)
    for i in range(n):
        out[i, i] = scalar
    return out


def matrix_padding(matrix: np.ndarray, *, pad_rows: bool = True) -> np.ndarray:
    m_row, n_col = matrix.shape
    target_rows = _ceil_block(m_row, BLOCK) if pad_rows else m_row
    target_cols = _ceil_block(n_col, BLOCK)
    return np.pad(
        matrix,
        ((0, target_rows - m_row), (0, target_cols - n_col)),
        mode="constant",
    )


def parse_qlinearmul(onnx_path: Path) -> Dict[str, Any]:
    model = onnx.load(str(onnx_path))
    node = next(n for n in model.graph.node if n.op_type == "QLinearMul")
    graph = model.graph

    def _scalar(name: str):
        init = next(t for t in graph.initializer if t.name == name)
        return numpy_helper.to_array(init).item()

    const_name = node.input[0]
    const_init = next(t for t in graph.initializer if t.name == const_name)
    scalar = int(numpy_helper.to_array(const_init).reshape(-1)[0])

    x_name = node.input[3]
    x_vi = next(v for v in graph.input if v.name == x_name)
    _, nc, nh, nw = [int(d.dim_value) for d in x_vi.type.tensor_type.shape.dim]
    m_rows = nh * nw

    return {
        "name": node.name or "QLinearMul_Node",
        "scalar": scalar,
        "input_shape": (1, nc, nh, nw),
        "output_shape": (1, nc, nh, nw),
        "offsetA": int(_scalar(node.input[5])),
        "scaleA": float(_scalar(node.input[4])),
        "offsetB": int(_scalar(node.input[2])),
        "scaleB": float(_scalar(node.input[1])),
        "offsetC": int(_scalar(node.input[7])),
        "scaleC": float(_scalar(node.input[6])),
        "m_rows": m_rows,
        "nc": nc,
    }


def build_standalone_16x16(name: str = "_T2", scalar: int = 1) -> Dict[str, Any]:
    rng = np.random.default_rng(7)
    inp = rng.integers(-128, 127, size=(16, 16), dtype=np.int32)
    return {
        "name": name,
        "scalar": scalar,
        "input_shape": (1, 16, 1, 16),
        "output_shape": (1, 16, 1, 16),
        "offsetA": 0,
        "scaleA": 1.0,
        "offsetB": 0,
        "scaleB": 1.0,
        "offsetC": 0,
        "scaleC": 1.0,
        "m_rows": 16,
        "nc": 16,
        "input_matrix": inp,
    }


def gemm_from_meta(meta: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    name = meta["name"]
    scalar = meta["scalar"]
    if "input_matrix" in meta:
        a_raw = meta["input_matrix"]
    else:
        a_raw = np.zeros((meta["m_rows"], meta["nc"]), dtype=np.int32)
    a_pad = matrix_padding(a_raw, pad_rows=True)
    b_pad = matrix_padding(matrix_diagonal(scalar, a_pad.shape[1], a_pad.shape[1]))
    m, k, n = a_pad.shape[0], a_pad.shape[1], b_pad.shape[1]

    raw_in = out_dir / f"input_{name}_{m}x{k}.bin"
    if m == 16 and k == 16 and name == "_T2":
        raw_in = out_dir / "input_16x16.bin"
    raw_wgt = out_dir / f"weight_{name}_{k}x{n}.bin"
    a_pad.astype(np.int32).tofile(raw_in)
    b_pad.astype(np.int32).tofile(raw_wgt)

    gemm_mlir = (
        f"  vta.gemm ins(%lhs_{name}, %rhs_{name} : memref<{m}x{k}xi32>, "
        f"memref<{k}x{n}xi32>)\n"
        f"           outs(%acc_{name} : memref<{m}x{n}xi32>) "
        f'{{name = "{name}", strategy = 1 : i64, mul_constant = {scalar} : i64}}\n'
    )
    func_args = (
        f"%lhs_{name}: memref<{m}x{k}xi32>, %rhs_{name}: memref<{k}x{n}xi32>, "
        f"%acc_{name}: memref<{m}x{n}xi32>"
    )
    return {
        "name": name,
        "processor": "vta",
        "reshape": "im2row",
        "offsetA": meta["offsetA"],
        "scaleA": meta["scaleA"],
        "offsetB": meta["offsetB"],
        "scaleB": meta["scaleB"],
        "offsetC": meta["offsetC"],
        "scaleC": meta["scaleC"],
        "rescaling": (meta["scaleA"] * meta["scaleB"]) / meta["scaleC"],
        "input_shape": list(meta["input_shape"]),
        "kernel": (1, 1),
        "stride": (1, 1),
        "padding": (0, 0, 0, 0),
        "output_shape": list(meta["output_shape"]),
        "parent_layers": ["image"],
        "raw_dims": {"M": m, "K": k, "N": n},
        "mul_constant": scalar,
        "paths": {"input": str(raw_in), "weight": str(raw_wgt)},
        "func_args": func_args,
        "gemm_mlir": gemm_mlir,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="QLinearMul → vta.gemm (mul_constant)")
    ap.add_argument("onnx", nargs="?", help="Optional qlinearmul_debug.onnx")
    ap.add_argument("-o", "--out", type=Path, default=Path("/tmp/onnx_qlinearmul"))
    ap.add_argument("--standalone-16", action="store_true", help="16x16 golden test")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.onnx:
        meta_src = parse_qlinearmul(Path(args.onnx))
    else:
        meta_src = build_standalone_16x16()

    layer = gemm_from_meta(meta_src, args.out)
    mlir_path = args.out / "qlinearmul.mlir"
    mlir_path.write_text(
        f"func @main({layer['func_args']}) {{\n{layer['gemm_mlir']}  return\n}}\n",
        encoding="utf-8",
    )
    layer["mlir"] = str(mlir_path)
    (args.out / "bridge_meta.json").write_text(
        json.dumps(layer, indent=2), encoding="utf-8"
    )
    print(f"MLIR: {mlir_path}  mul_constant={layer['mul_constant']}")


if __name__ == "__main__":
    main()
