#!/usr/bin/env python3
"""Bridge ReLU ONNX (or fixed 16x16 test) to vta.alu MLIR + accumulator bin.

Increment E: maps upstream Relu semantics (MAX_IMM 0 over NCHW flattened rows)
to vta.alu {alu_opcode=1, use_imm=true, imm=0}.

Supports:
  - Fixed 16x16 standalone test (default, matches relu_16x16 golden)
  - ONNX with a single Relu node after QLinearConv (--onnx conv_relu_debug.onnx)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx

BLOCK = 16
SVTA = Path("/mnt/c/MLIR-VTA/standalone-vta")


def _ceil_block(n: int, block: int = BLOCK) -> int:
    return ((n - 1) // block + 1) * block


def fixed_accumulator_16x16() -> np.ndarray:
    """Same pattern as make_golden_relu_upstream.sh for byte-level compare."""
    rows = [
        [-5, 3, -1, 7, 0, -9, 2, 4, -2, 1, 6, -8, 3, -4, 5, 0],
        [1] * 16,
        [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ]
    rows += [[(-1) ** (i + j) * (i + j) for j in range(16)] for i in range(3, 16)]
    return np.array(rows, dtype=np.int32)


def pad_matrix_to_blocked(acc: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pad [M, N] to [M, 16] with zero columns; return (padded, M)."""
    m, n = acc.shape
    if n > BLOCK:
        raise ValueError(f"Relu bridge supports N <= {BLOCK}, got {n}")
    out = np.zeros((m, BLOCK), dtype=np.int32)
    out[:, :n] = acc
    return out, m


def to_blocked_bin(acc: np.ndarray) -> bytes:
    """Row-major 16x16 blocks (one block when M<=16 and cols=16)."""
    m, n = acc.shape
    mb = _ceil_block(m, BLOCK) // BLOCK
    nb = n // BLOCK
    out: List[int] = []
    for bi in range(mb):
        for bj in range(nb):
            for r in range(BLOCK):
                for c in range(BLOCK):
                    gr, gc = bi * BLOCK + r, bj * BLOCK + c
                    v = int(acc[gr, gc]) if gr < m and gc < n else 0
                    out.append(v)
    return np.array(out, dtype=np.int32).tobytes()


def parse_relu_after_conv(onnx_path: Path) -> Dict[str, Any]:
    model = onnx.load(str(onnx_path))
    conv = relu = None
    for node in model.graph.node:
        if node.op_type == "QLinearConv":
            conv = node
        elif node.op_type == "Relu":
            relu = node
    if conv is None or relu is None:
        raise SystemExit(f"Expected QLinearConv -> Relu in {onnx_path}")
    if relu.input[0] != conv.output[0]:
        raise SystemExit("Relu input must be QLinearConv output")

    out_vi = next(v for v in model.graph.value_info if v.name == conv.output[0])
    _, oc, oh, ow = [int(d) for d in out_vi.type.tensor_type.shape.dim]
    m_rows = oh * ow
    acc, m = pad_matrix_to_blocked(np.zeros((m_rows, oc), dtype=np.int32))
    return {
        "name": relu.name or "Relu1",
        "M": m,
        "output_shape": [1, oc, oh, ow],
        "parent_layers": [conv.name or "QLinearConv1"],
        "accumulator": acc,
    }


def build_standalone_16x16(name: str = "Relu1") -> Dict[str, Any]:
    acc = fixed_accumulator_16x16()
    return {
        "name": name,
        "M": 16,
        "output_shape": [1, 16, 1, 16],
        "parent_layers": ["image"],
        "accumulator": acc,
    }


def emit_mlir(meta: Dict[str, Any]) -> str:
    m = meta["M"]
    name = meta["name"]
    return (
        f"func @main(%acc_{name}: memref<{m}x{BLOCK}xi32>) {{\n"
        f"  vta.alu ins(%acc_{name} : memref<{m}x{BLOCK}xi32>) "
        f"{{name = \"{name}\", alu_opcode = 1 : i64, use_imm = true, imm = 0 : i64}}\n"
        f"  return\n}}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX ReLU → vta.alu bridge")
    ap.add_argument("onnx", nargs="?", help="Optional ONNX with Conv+Relu")
    ap.add_argument("-o", "--out", type=Path, default=Path("/tmp/onnx_relu"))
    ap.add_argument("--standalone-16", action="store_true", default=False,
                    help="16x16 fixed test (default when onnx omitted)")
    args = ap.parse_args()

    if args.onnx:
        meta_src = parse_relu_after_conv(Path(args.onnx))
    else:
        meta_src = build_standalone_16x16()

    args.out.mkdir(parents=True, exist_ok=True)
    name = meta_src["name"]
    acc = meta_src["accumulator"]
    raw_path = args.out / f"accumulator_{acc.shape[0]}x{acc.shape[1]}.bin"
    acc.tofile(raw_path)
    blocked_path = args.out / "accumulator_blocked.bin"
    blocked_path.write_bytes(to_blocked_bin(acc))

    mlir_path = args.out / "relu.mlir"
    mlir_path.write_text(emit_mlir(meta_src), encoding="utf-8")

    meta = {
        "name": name,
        "M": meta_src["M"],
        "processor": "vta",
        "reshape": "int32",
        "offsetA": 0,
        "scaleA": 1.0,
        "offsetB": 0,
        "scaleB": 1.0,
        "offsetC": 0,
        "scaleC": 1.0,
        "rescaling": 1.0,
        "input_shape": meta_src.get("input_shape", [1, 16, 1, 16]),
        "kernel": (1, 1),
        "stride": (1, 1),
        "padding": (0, 0, 0, 0),
        "output_shape": meta_src["output_shape"],
        "parent_layers": meta_src["parent_layers"],
        "alu_opcode": 1,
        "use_imm": True,
        "imm": 0,
        "paths": {
            "accumulator_raw": str(raw_path),
            "accumulator_blocked": str(blocked_path),
        },
        "mlir": str(mlir_path),
    }
    (args.out / "bridge_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"MLIR: {mlir_path}  M={meta['M']} MAX_IMM relu")


if __name__ == "__main__":
    main()
