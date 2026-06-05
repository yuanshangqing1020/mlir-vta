#!/usr/bin/env python3
"""Bridge a single-layer QLinearConv ONNX model to padded GEMM MLIR + raw bins.

Uses the `onnx` package only (no onnxruntime). Padding rules follow
standalone-vta data_definition.matrix_padding; rows are additionally
padded to 16-multiples for mlir-vta vta.gemm verifier (increment A).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

BLOCK = 16
SVTA = Path("/mnt/c/MLIR-VTA/standalone-vta")
sys.path.insert(0, str(SVTA / "src" / "compiler"))

import nn_compiler.shape_data.shape_data as SD  # noqa: E402
from utils import tensor_matrix_converter as TM  # noqa: E402


def _ceil_block(n: int, block: int = BLOCK) -> int:
    return ((n - 1) // block + 1) * block


def matrix_padding(
    matrix: np.ndarray,
    *,
    is_weight: bool = False,
    is_square: bool = True,
    pad_rows: bool = True,
    block: int = BLOCK,
) -> np.ndarray:
    """Pad matrix per upstream data_definition.py (+ optional row pad for MLIR)."""
    m_row, n_col = matrix.shape
    if is_weight or (is_square and pad_rows):
        target_rows = _ceil_block(m_row, block)
    else:
        target_rows = _ceil_block(m_row, block) if pad_rows else m_row
    target_cols = _ceil_block(n_col, block)
    return np.pad(
        matrix,
        ((0, target_rows - m_row), (0, target_cols - n_col)),
        mode="constant",
    )


def im2row(
    x: np.ndarray,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int, int, int],
    dtype=np.int8,
) -> np.ndarray:
    """NCHW batch=1 im2row → (mh*mw, nc*fh*fw)."""
    kh, kw = kernel_size
    sh, sw = stride
    pt, pl, pb, pr = padding
    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (pt, pb), (pl, pr)),
        mode="constant",
        constant_values=0,
    )
    _, nc, ih, iw = x_pad.shape
    oh = (ih - kh) // sh + 1
    ow = (iw - kw) // sw + 1
    rows = oh * ow
    cols = nc * kh * kw
    out = np.zeros((rows, cols), dtype=dtype)
    r = 0
    for i in range(0, ih - kh + 1, sh):
        for j in range(0, iw - kw + 1, sw):
            patch = x_pad[0, :, i : i + kh, j : j + kw]
            out[r] = patch.reshape(-1)
            r += 1
    return out


def parse_qlinearconv(onnx_path: Path) -> Dict[str, Any]:
    model = onnx.load(str(onnx_path))
    graph = model.graph
    conv = next(n for n in graph.node if n.op_type == "QLinearConv")

    def _scalar(name: str):
        init = next(t for t in graph.initializer if t.name == name)
        arr = numpy_helper.to_array(init)
        return arr.item() if arr.ndim == 0 else arr

    w_init = next(t for t in graph.initializer if t.name == "W")
    w = numpy_helper.to_array(w_init).astype(np.int8)
    b = numpy_helper.to_array(
        next(t for t in graph.initializer if t.name == "B")
    ).astype(np.int32)

    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in conv.attribute}
    pads = list(attrs.get("pads", [0, 0, 0, 0]))
    strides = list(attrs.get("strides", [1, 1]))
    kernel = list(attrs.get("kernel_shape", [3, 3]))

    x_info = next(vi for vi in graph.input if vi.name == "X")
    y_info = next(vi for vi in graph.output if vi.name == "Y")
    _, nc, nh, nw = [d.dim_value for d in x_info.type.tensor_type.shape.dim]
    _, mc, mh, mw = [d.dim_value for d in y_info.type.tensor_type.shape.dim]

    return {
        "name": conv.name or "QLinearConv1",
        "input_shape": (1, nc, nh, nw),
        "output_shape": (1, mc, mh, mw),
        "kernel": tuple(kernel),
        "stride": tuple(strides),
        "padding": (pads[0], pads[1], pads[2], pads[3]),
        "weight": w,
        "bias": b,
        "offsetA": int(_scalar("x_zp")),
        "scaleA": float(_scalar("x_scale")),
        "offsetB": int(_scalar("w_zp")),
        "scaleB": float(_scalar("w_scale")),
        "offsetC": int(_scalar("y_zp")),
        "scaleC": float(_scalar("y_scale")),
    }


def build_layer(
    onnx_path: Path,
    out_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    info = parse_qlinearconv(onnx_path)
    name = info["name"]
    nc, nh, nw = info["input_shape"][1:]
    mc, mh, mw = info["output_shape"][1:]
    fh, fw = info["kernel"]
    sh, sw = info["stride"]
    ph = (info["padding"][0], info["padding"][2])
    pw = (info["padding"][1], info["padding"][3])

    (ah, aw), (bh, bw), (ch, cw) = TM.im2row_matrix_dimension(
        nc=nc,
        nh=nh,
        nw=nw,
        mc=mc,
        mh=mh,
        mw=mw,
        fh=fh,
        fw=fw,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
    )
    assert ah == ch and aw == bh and bw == mc

    np.random.seed(seed)
    x_nchw = np.random.randint(-128, 127, size=(1, nc, nh, nw), dtype=np.int8)
    a_raw = im2row(
        x_nchw,
        (fh, fw),
        (sh, sw),
        info["padding"],
        dtype=np.int32,
    )
    wgt = info["weight"]
    if info["offsetB"] != 0:
        wgt = wgt - info["offsetB"]
    b_raw = SD.ker2col(wgt, dtype=np.int32)
    bias = info["bias"].reshape(1, -1).astype(np.int32)
    acc_raw = np.tile(bias, (ah, 1))

    a_pad = matrix_padding(a_raw, is_square=False, pad_rows=True)
    b_pad = matrix_padding(b_raw, is_weight=True, pad_rows=True)
    acc_pad = matrix_padding(acc_raw, is_weight=True, pad_rows=True)

    m, k, n = a_pad.shape[0], a_pad.shape[1], b_pad.shape[1]
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_in = out_dir / f"input_{m}x{k}.bin"
    raw_wgt = out_dir / f"weight_{k}x{n}.bin"
    raw_acc = out_dir / f"accumulator_{m}x{n}.bin"
    a_pad.astype(np.int32).tofile(raw_in)
    b_pad.astype(np.int32).tofile(raw_wgt)
    acc_pad.astype(np.int32).tofile(raw_acc)

    mlir_path = out_dir / "qlinearconv_padded.mlir"
    mlir_text = f"""// Padded QLinearConv GEMM (increment A): {ah}x{aw} x {aw}x{bw} -> {ah}x{bw}
// mlir dims {m}x{k}x{n} (rows padded to 16-multiple for vta.gemm verifier)
func @main(%lhs: memref<{m}x{k}xi32>, %rhs: memref<{k}x{n}xi32>, %acc: memref<{m}x{n}xi32>) {{
  vta.gemm ins(%lhs, %rhs : memref<{m}x{k}xi32>, memref<{k}x{n}xi32>)
           outs(%acc : memref<{m}x{n}xi32>) {{name = "{name}", strategy = 2 : i64}}
  return
}}
"""
    mlir_path.write_text(mlir_text, encoding="utf-8")

    meta = {
        "name": name,
        "processor": "vta",
        "reshape": "im2row",
        "offsetA": info["offsetA"],
        "scaleA": info["scaleA"],
        "offsetB": info["offsetB"],
        "scaleB": info["scaleB"],
        "offsetC": info["offsetC"],
        "scaleC": info["scaleC"],
        "rescaling": (info["scaleA"] * info["scaleB"]) / info["scaleC"],
        "input_shape": info["input_shape"],
        "kernel": info["kernel"],
        "stride": info["stride"],
        "padding": info["padding"],
        "output_shape": info["output_shape"],
        "parent_layers": ["image"],
        "raw_dims": {"M": m, "K": k, "N": n, "ah": ah, "aw": aw},
        "paths": {
            "input": str(raw_in),
            "weight": str(raw_wgt),
            "accumulator": str(raw_acc),
            "mlir": str(mlir_path),
        },
        "input_nchw_int8": x_nchw,
        "a_raw": a_raw,
        "b_raw": b_raw,
        "acc_raw": acc_raw,
    }
    (out_dir / "bridge_meta.json").write_text(
        json.dumps({k: v for k, v in meta.items() if k not in ("input_nchw_int8", "a_raw", "b_raw", "acc_raw")},
                   indent=2),
        encoding="utf-8",
    )
    np.savez(
        out_dir / "bridge_arrays.npz",
        x_nchw=x_nchw,
        a_raw=a_raw,
        b_raw=b_raw,
        acc_raw=acc_raw,
        a_pad=a_pad,
        b_pad=b_pad,
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "onnx",
        type=Path,
        default=SVTA / "examples/onnx/qlinearconv_debug.onnx",
        nargs="?",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("/tmp/onnx_qconv_bridge"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = build_layer(args.onnx, args.out_dir, seed=args.seed)
    d = meta["raw_dims"]
    print(
        f"Bridge OK: {meta['name']} padded GEMM {d['M']}x{d['K']}x{d['N']} "
        f"(logical {d['ah']}x{d['aw']}x{meta['output_shape'][1]})"
    )
    print(f"  MLIR: {meta['paths']['mlir']}")


if __name__ == "__main__":
    main()
