#!/usr/bin/env python3
"""Reference for qadd_two_mul (FSIM im2row + mul_constant + qadd)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MLIRVTA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MLIRVTA / "scripts"))
from fsim_reference_utils import qadd_two_mul_reference, write_bins  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-matrix", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--scalar1", type=int, default=2)
    ap.add_argument("--scalar2", type=int, default=3)
    args = ap.parse_args()

    a = np.fromfile(args.input_matrix, dtype=np.int32).reshape(16, 16)
    inp, ref = qadd_two_mul_reference(a, scalar1=args.scalar1, scalar2=args.scalar2)
    write_bins(args.out, inp, ref)
    print(f"Wrote {args.out}/input_nn.bin and reference.bin (FSIM-aligned)")


if __name__ == "__main__":
    main()
