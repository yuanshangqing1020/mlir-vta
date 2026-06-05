#!/usr/bin/env python3
"""Emit dependency.csv for fsim_nn from layer metadata.

Format matches standalone-vta vta_backend.py output; see
standalone-vta/docs/dependency_csv详解.md.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Sequence, Tuple


def _crlf_lines(rows: Sequence[Sequence[Any]]) -> str:
    return "\r\n".join(",".join(str(c) for c in row) for row in rows) + "\r\n"


def write_dependency_csv(
    path: str,
    layers: List[Dict[str, Any]],
    *,
    image_rows: int,
    image_cols: int,
    output_layer: str,
    output_nchw: Tuple[int, int, int],
) -> None:
    """Write a single-network dependency.csv.

    Each layer dict must include keys used by vta_backend node_info:
    offsetA, scaleA, offsetB, scaleB, offsetC, scaleC, rescaling,
    input_shape (N,C,H,W), kernel (fh,fw), stride (sh,sw),
    padding (pt, pl, pb, pr), output_shape (N,C,H,W), parent_layers (list).
    """
    nb_steps = len(layers)
    oc, oh, ow = output_nchw

    rows: List[Tuple[Any, ...]] = [
        ("Line identifier", "Number of layers"),
        ("nb_steps", nb_steps),
        ("Line identifier", "Nb rows", "Nb columns"),
        ("image", image_rows, image_cols),
        (
            "Line identifier",
            "Final layer name",
            "Output tensor channels",
            "Output tensor height",
            "Output tensor width",
        ),
        ("output", output_layer, oc, oh, ow),
        ("Execution order", "Processor", "Layer name"),
    ]
    for i, layer in enumerate(layers):
        rows.append((i, layer.get("processor", "vta"), layer["name"]))

    rows.append(
        (
            "Layer name",
            "Processor",
            "Reshape",
            "offsetA",
            "scaleA",
            "offsetB",
            "scaleB",
            "offsetU",
            "scaleU",
            "offsetV",
            "scaleV",
            "Input channels",
            "Input height",
            "Input width",
            "Kernel height",
            "Kernel width",
            "Stride height",
            "Stride width",
            "Padding top",
            "Padding left",
            "Padding bottom",
            "Padding right",
            "Output channel",
            "Output height",
            "Output width",
            "offsetC",
            "scaleC",
            "Rescaling factor",
            "Parent layers",
        )
    )

    for layer in layers:
        n, c, h, w = layer["input_shape"]
        oc_l, oh_l, ow_l = layer["output_shape"][1:]
        fh, fw = layer["kernel"]
        sh, sw = layer["stride"]
        pt, pl, pb, pr = layer["padding"]
        parents: List[str] = layer.get("parent_layers", ["image"])
        nb_inp = len(parents)
        row: List[Any] = [
            layer["name"],
            layer.get("processor", "vta"),
            layer.get("reshape", "im2row"),
            layer["offsetA"],
            layer["scaleA"],
            layer["offsetB"],
            layer["scaleB"],
            0,
            1,
            0,
            1,
            c,
            h,
            w,
            fh,
            fw,
            sh,
            sw,
            pt,
            pl,
            pb,
            pr,
            oc_l,
            oh_l,
            ow_l,
            layer["offsetC"],
            layer["scaleC"],
            layer["rescaling"],
            "INP",
            nb_inp,
        ]
        row.extend(parents)
        rows.append(tuple(row))

    with open(path, "w", newline="") as f:
        f.write(_crlf_lines(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit dependency.csv for tests")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--layer-name", default="QLinearConv1")
    parser.add_argument("--image-rows", type=int, default=25)
    parser.add_argument("--image-cols", type=int, default=3)
    args = parser.parse_args()

    layer = {
        "name": args.layer_name,
        "processor": "vta",
        "reshape": "im2row",
        "offsetA": -128,
        "scaleA": 0.031500000506639481,
        "offsetB": 0,
        "scaleB": 0.004999999888241291,
        "offsetC": -128,
        "scaleC": 0.024100000038743019,
        "rescaling": 0.0065352697856724262,
        "input_shape": (1, 3, 5, 5),
        "kernel": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1, 1, 1),
        "output_shape": (1, 3, 5, 5),
        "parent_layers": ["image"],
    }
    write_dependency_csv(
        args.output,
        [layer],
        image_rows=args.image_rows,
        image_cols=args.image_cols,
        output_layer=args.layer_name,
        output_nchw=(3, 5, 5),
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
