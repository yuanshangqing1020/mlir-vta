#!/usr/bin/env python3
"""FSIM-aligned reference helpers (im2row + flat/NCHW conventions)."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "standalone-vta" / "src" / "compiler"))
from reference_computation.reference_onnx import im2row  # noqa: E402


def flat_matrix_to_nchw(flat: np.ndarray, channels: int, height: int, width: int) -> np.ndarray:
    """Map 16×16-style flat (rows=h*w, cols=c) to NCHW [1,C,H,W]."""
    m = flat.reshape(height * width, channels)
    x = np.zeros((1, channels, height, width), dtype=np.int8)
    for h in range(height):
        for w in range(width):
            row = h * width + w
            for c in range(channels):
                x[0, c, h, w] = m[row, c]
    return x


def nchw_to_flat_matrix(x: np.ndarray) -> np.ndarray:
    """Inverse of flat_matrix_to_nchw for [1,C,H,W]."""
    _, c, h, w = x.shape
    m = np.zeros((h * w, c), dtype=x.dtype)
    for hi in range(h):
        for wi in range(w):
            row = hi * w + wi
            for ci in range(c):
                m[row, ci] = x[0, ci, hi, wi]
    return m.reshape(-1)


def input_matrix_to_input_nn(matrix: np.ndarray) -> np.ndarray:
    """Match run_fsim_nn_2layer / fsim reshape input vector."""
    return matrix.astype(np.int8).reshape(-1)


def qadd_two_mul_reference(
    matrix: np.ndarray,
    *,
    scalar1: int = 2,
    scalar2: int = 3,
    channels: int = 16,
    height: int = 1,
    width: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (input_nn, reference_nchw_flat) for dual mul_constant + qadd."""
    flat = input_matrix_to_input_nn(matrix)
    x = flat_matrix_to_nchw(flat, channels, height, width)
    m = im2row(x.astype(np.int32), kernel_size=(1, 1), stride=(1, 1), padding=(0, 0, 0, 0))
    y1 = np.clip(m * scalar1, -128, 127)
    y2 = np.clip(m * scalar2, -128, 127)
    acc = np.clip(y1 + y2, -128, 127).astype(np.int8)
    ref = np.zeros((1, channels, height, width), dtype=np.int8)
    for hi in range(height):
        for wi in range(width):
            row = hi * width + wi
            for c in range(channels):
                ref[0, c, hi, wi] = acc[row, c]
    return flat, ref.reshape(-1)


def write_bins(
    out_dir: Path,
    input_nn: np.ndarray,
    reference: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    input_nn.astype(np.int8).tofile(out_dir / "input_nn.bin")
    reference.astype(np.int8).tofile(out_dir / "reference.bin")
