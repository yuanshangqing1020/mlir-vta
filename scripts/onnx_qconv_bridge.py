#!/usr/bin/env python3
"""Bridge QLinearConv ONNX model(s) to GEMM MLIR + raw bins.

Uses the `onnx` package only. Supports:
  --layout col-pad   : upstream-style column padding (increment B, M may be non-16)
  --layout square-pad: row+col pad to 16-multiples (increment A legacy)
  --multi            : emit all QLinearConv nodes in one func (increment C)
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


def parse_qlinearconv_node(model: onnx.ModelProto, node) -> Dict[str, Any]:
    graph = model.graph

    def _scalar(name: str):
        init = next(t for t in graph.initializer if t.name == name)
        arr = numpy_helper.to_array(init)
        return arr.item() if arr.ndim == 0 else arr

    w_name = node.input[3]
    w_init = next(t for t in graph.initializer if t.name == w_name)
    w = numpy_helper.to_array(w_init).astype(np.int8)
    b_name = node.input[8]
    b = numpy_helper.to_array(
        next(t for t in graph.initializer if t.name == b_name)
    ).astype(np.int32)

    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    pads = list(attrs.get("pads", [0, 0, 0, 0]))
    strides = list(attrs.get("strides", [1, 1]))
    kernel = list(attrs.get("kernel_shape", [3, 3]))

    x_name = node.input[0]
    y_name = node.output[0]

    def _tensor_info(name: str):
        for src in (graph.input, graph.output, graph.value_info):
            for vi in src:
                if vi.name == name:
                    return vi
        return None

    x_info = _tensor_info(x_name)
    if x_info is None:
        raise KeyError(f"tensor {x_name!r} not in graph inputs/outputs/value_info")
    _, nc, nh, nw = [d.dim_value for d in x_info.type.tensor_type.shape.dim]

    y_info = _tensor_info(y_name)
    if y_info is None:
        kh, kw = kernel[0], kernel[1]
        sh, sw = strides[0], strides[1]
        pt, pl, pb, pr = pads[0], pads[1], pads[2], pads[3]
        mh = (nh + pt + pb - kh) // sh + 1
        mw = (nw + pl + pr - kw) // sw + 1
        mc = w.shape[0]
        output_shape = (1, mc, mh, mw)
    else:
        output_shape = tuple(d.dim_value for d in y_info.type.tensor_type.shape.dim)
    _, mc, mh, mw = output_shape

    x_scale = node.input[1]
    x_zp = node.input[2]
    w_scale = node.input[4]
    w_zp = node.input[5]
    y_scale = node.input[6]
    y_zp = node.input[7]

    return {
        "name": node.name or node.output[0].replace("/", "_"),
        "input_shape": (1, nc, nh, nw),
        "output_shape": output_shape,
        "kernel": tuple(kernel),
        "stride": tuple(strides),
        "padding": (pads[0], pads[1], pads[2], pads[3]),
        "weight": w,
        "bias": b,
        "offsetA": int(_scalar(x_zp)),
        "scaleA": float(_scalar(x_scale)),
        "offsetB": int(_scalar(w_zp)),
        "scaleB": float(_scalar(w_scale)),
        "offsetC": int(_scalar(y_zp)),
        "scaleC": float(_scalar(y_scale)),
    }


def list_qlinearconv_nodes(onnx_path: Path) -> List:
    model = onnx.load(str(onnx_path))
    return [n for n in model.graph.node if n.op_type == "QLinearConv"], model


def gemm_from_info(
    info: Dict[str, Any],
    x_nchw: np.ndarray,
    layout: str,
    out_dir: Path,
) -> Dict[str, Any]:
    name = info["name"]
    nc, nh, nw = info["input_shape"][1:]
    mc, mh, mw = info["output_shape"][1:]
    fh, fw = info["kernel"]
    sh, sw = info["stride"]
    ph = (info["padding"][0], info["padding"][2])
    pw = (info["padding"][1], info["padding"][3])

    (ah, aw), (bh, bw), (_, _) = TM.im2row_matrix_dimension(
        nc=nc, nh=nh, nw=nw, mc=mc, mh=mh, mw=mw,
        fh=fh, fw=fw, sh=sh, sw=sw, ph=ph, pw=pw,
    )

    a_raw = im2row(x_nchw, (fh, fw), (sh, sw), info["padding"], dtype=np.int32)
    wgt = info["weight"]
    if info["offsetB"] != 0:
        wgt = wgt - info["offsetB"]
    b_raw = SD.ker2col(wgt, dtype=np.int32)
    bias = info["bias"].reshape(1, -1).astype(np.int32)

    col_pad = layout == "col-pad"
    a_pad = matrix_padding(a_raw, is_square=False, pad_rows=not col_pad)
    b_pad = matrix_padding(b_raw, is_weight=True, pad_rows=True)
    if col_pad:
        acc_pad = matrix_padding(bias, is_weight=False, is_square=False, pad_rows=False)
        expand_bias = True
    else:
        acc_full = np.tile(bias, (ah, 1))
        acc_pad = matrix_padding(acc_full, is_weight=True, pad_rows=True)
        expand_bias = False

    m, k, n = a_pad.shape[0], a_pad.shape[1], b_pad.shape[1]
    raw_in = out_dir / f"input_{name}_{m}x{k}.bin"
    raw_wgt = out_dir / f"weight_{name}_{k}x{n}.bin"
    raw_acc = out_dir / f"accumulator_{name}_{acc_pad.shape[0]}x{acc_pad.shape[1]}.bin"
    a_pad.astype(np.int32).tofile(raw_in)
    b_pad.astype(np.int32).tofile(raw_wgt)
    acc_pad.astype(np.int32).tofile(raw_acc)

    expand_attr = ", expand_bias = true" if expand_bias else ""
    gemm_mlir = (
        f"  vta.gemm ins(%lhs_{name}, %rhs_{name} : memref<{m}x{k}xi32>, "
        f"memref<{k}x{n}xi32>)\n"
        f"           outs(%acc_{name} : memref<{m}x{n}xi32>) "
        f"{{name = \"{name}\", strategy = 2 : i64{expand_attr}}}\n"
    )
    func_args = (
        f"%lhs_{name}: memref<{m}x{k}xi32>, %rhs_{name}: memref<{k}x{n}xi32>, "
        f"%acc_{name}: memref<{m}x{n}xi32>"
    )

    node_info = {
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
        "expand_bias": expand_bias,
        "acc_rows": acc_pad.shape[0],
        "paths": {
            "input": str(raw_in),
            "weight": str(raw_wgt),
            "accumulator": str(raw_acc),
        },
        "func_args": func_args,
        "gemm_mlir": gemm_mlir,
    }
    return node_info


def build_model(
    onnx_path: Path,
    out_dir: Path,
    seed: int = 42,
    layout: str = "col-pad",
    multi: bool = False,
    zero_input: bool = False,
) -> Dict[str, Any]:
    nodes, model = list_qlinearconv_nodes(onnx_path)
    if not nodes:
        raise SystemExit(f"No QLinearConv in {onnx_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    layers: List[Dict[str, Any]] = []
    parent_map = {"image": None}

    for idx, node in enumerate(nodes):
        info = parse_qlinearconv_node(model, node)
        if idx == 0:
            nc, nh, nw = info["input_shape"][1:]
            if zero_input:
                x_nchw = np.zeros((1, nc, nh, nw), dtype=np.int8)
            else:
                x_nchw = np.random.randint(-128, 127, size=(1, nc, nh, nw), dtype=np.int8)
        else:
            # Placeholder int8 tensor for layer>0 (fsim_nn chains real outputs).
            nc, nh, nw = info["input_shape"][1:]
            x_nchw = np.zeros((1, nc, nh, nw), dtype=np.int8)

        meta = gemm_from_info(info, x_nchw, layout, out_dir)
        if idx == 0:
            meta["parent_layers"] = ["image"]
        else:
            prev = layers[-1]["name"]
            meta["parent_layers"] = [prev]
        layers.append(meta)
        x_nchw.tofile(out_dir / f"input_nchw_{meta['name']}.bin")

    if multi and len(layers) > 1:
        args = ", ".join(l["func_args"] for l in layers)
        body = "".join(l["gemm_mlir"] for l in layers)
        mlir_text = f"func @main({args}) {{\n{body}  return\n}}\n"
        mlir_name = "qlinearconv_multi.mlir"
    else:
        l = layers[0]
        mlir_text = (
            f"func @main({l['func_args']}) {{\n{l['gemm_mlir']}  return\n}}\n"
        )
        mlir_name = (
            "qlinearconv_colpad.mlir" if layout == "col-pad" else "qlinearconv_padded.mlir"
        )

    mlir_path = out_dir / mlir_name
    mlir_path.write_text(mlir_text, encoding="utf-8")

    result = {
        "layers": layers,
        "mlir": str(mlir_path),
        "layout": layout,
        "multi": multi,
    }
    meta_json = {k: v for k, v in result.items() if k != "layers"}
    meta_json["layers"] = [
        {kk: vv for kk, vv in l.items() if kk not in ("gemm_mlir", "func_args")}
        for l in layers
    ]
    (out_dir / "bridge_meta.json").write_text(
        json.dumps(meta_json, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path, nargs="?", default=SVTA / "examples/onnx/qlinearconv_debug.onnx")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("/tmp/onnx_qconv_bridge"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layout",
        choices=("col-pad", "square-pad"),
        default="col-pad",
    )
    parser.add_argument("--multi", action="store_true")
    parser.add_argument(
        "--zero-input",
        action="store_true",
        help="Use all-zero int8 input (matches upstream golden generation)",
    )
    args = parser.parse_args()

    meta = build_model(args.onnx, args.out_dir, seed=args.seed,
                       layout=args.layout, multi=args.multi,
                       zero_input=args.zero_input)
    for l in meta["layers"]:
        d = l["raw_dims"]
        print(
            f"  {l['name']}: {d['M']}x{d['K']}x{d['N']} "
            f"(logical {d['ah']}x{d['aw']}x{l['output_shape'][1]}) "
            f"expand_bias={l['expand_bias']}"
        )
    print(f"MLIR: {meta['mlir']}")


if __name__ == "__main__":
    main()
