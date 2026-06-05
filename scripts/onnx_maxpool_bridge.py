#!/usr/bin/env python3
"""Bridge MaxPool (36x16 golden) to vta.maxpool MLIR + accumulator bin."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

BLOCK = 16
SVTA = Path("/mnt/c/MLIR-VTA/standalone-vta")


def fixed_accumulator_36x16() -> np.ndarray:
    """Reproducible pattern for maxpool_36x16 golden."""
    rng = np.random.default_rng(42)
    acc = rng.integers(-100, 100, size=(36, 16), dtype=np.int32)
    return acc


def to_blocked_bin(acc: np.ndarray) -> bytes:
    m, n = acc.shape
    mb = (m + BLOCK - 1) // BLOCK
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


def emit_mlir(name: str = "_T3", m: int = 36) -> str:
    return (
        f"func @main(%acc_{name}: memref<{m}x{BLOCK}xi32>) {{\n"
        f"  vta.maxpool ins(%acc_{name} : memref<{m}x{BLOCK}xi32>) "
        f'{{name = "{name}"}}\n'
        f"  return\n}}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="MaxPool 36x16 → vta.maxpool bridge")
    ap.add_argument("-o", "--out", type=Path, default=Path("/tmp/onnx_maxpool"))
    ap.add_argument("--name", default="_T3")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    acc = fixed_accumulator_36x16()
    raw_path = args.out / "accumulator_36x16.bin"
    acc.tofile(raw_path)
    blocked_path = args.out / "accumulator_blocked.bin"
    blocked_path.write_bytes(to_blocked_bin(acc))

    mlir_path = args.out / "maxpool.mlir"
    mlir_path.write_text(emit_mlir(args.name), encoding="utf-8")

    meta: Dict[str, Any] = {
        "name": args.name,
        "M": 36,
        "processor": "vta",
        "reshape": "int32",
        "offsetA": 0,
        "scaleA": 1.0,
        "offsetB": 0,
        "scaleB": 1.0,
        "offsetC": 0,
        "scaleC": 1.0,
        "rescaling": 1.0,
        "input_shape": [1, 16, 6, 6],
        "kernel": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0, 0, 0),
        "output_shape": [1, 16, 3, 3],
        "parent_layers": ["image"],
        "paths": {
            "accumulator_raw": str(raw_path),
            "accumulator_blocked": str(blocked_path),
        },
        "mlir": str(mlir_path),
    }
    (args.out / "bridge_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"MLIR: {mlir_path}  M=36 maxpool")


if __name__ == "__main__":
    main()
