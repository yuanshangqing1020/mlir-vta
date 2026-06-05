#!/usr/bin/env python3
"""Generate qconcat_two_branch.onnx: shared input, two QLinearConv, QLinearConcat (F-b)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = (
    Path(__file__).resolve().parents[2]
    / "standalone-vta"
    / "examples"
    / "onnx"
    / "qconcat_two_branch.onnx"
)


def scalar(name: str, val, dtype):
    return helper.make_tensor(name, dtype, [], [val])


def conv_weights(name_w, name_b, in_c: int, out_c: int, k: int = 3):
    w_shape = (out_c, in_c, k, k)
    np.random.seed(hash(name_w) & 0xFFFF)
    w = np.random.randint(-128, 127, size=w_shape, dtype=np.int8)
    b = np.random.randint(-500, 500, size=(out_c,), dtype=np.int32)
    return (
        helper.make_tensor(name_w, TensorProto.INT8, w_shape, w.tobytes(), raw=True),
        helper.make_tensor(name_b, TensorProto.INT32, [out_c], b.tobytes(), raw=True),
    )


def main() -> None:
    np.random.seed(42)
    in_c, out_c_each = 3, 4
    h, w = 5, 5
    k, pads, strides = 3, [1, 1, 1, 1], [1, 1]
    out_h = (h + pads[0] + pads[2] - k) // strides[0] + 1
    out_w = (w + pads[1] + pads[3] - k) // strides[1] + 1
    spatial = (out_h, out_w)
    out_c = out_c_each * 2

    w1, b1 = conv_weights("W1", "B1", in_c, out_c_each)
    w2, b2 = conv_weights("W2", "B2", in_c, out_c_each)

    initializers = [
        scalar("x_scale", 0.0315, TensorProto.FLOAT),
        scalar("x_zp", -128, TensorProto.INT8),
        w1,
        scalar("w1_scale", 0.0050, TensorProto.FLOAT),
        scalar("w1_zp", 0, TensorProto.INT8),
        scalar("y1_scale", 0.0241, TensorProto.FLOAT),
        scalar("y1_zp", -128, TensorProto.INT8),
        b1,
        w2,
        scalar("w2_scale", 0.0048, TensorProto.FLOAT),
        scalar("w2_zp", 0, TensorProto.INT8),
        scalar("y2_scale", 0.0241, TensorProto.FLOAT),
        scalar("y2_zp", -128, TensorProto.INT8),
        b2,
        scalar("y_scale", 0.0241, TensorProto.FLOAT),
        scalar("y_zp", -128, TensorProto.INT8),
    ]

    X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, in_c, h, w])
    Y1 = helper.make_tensor_value_info("Y1", TensorProto.INT8, [1, out_c_each, *spatial])
    Y2 = helper.make_tensor_value_info("Y2", TensorProto.INT8, [1, out_c_each, *spatial])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT8, [1, out_c, *spatial])

    conv1 = helper.make_node(
        "QLinearConv",
        ["X", "x_scale", "x_zp", "W1", "w1_scale", "w1_zp", "y1_scale", "y1_zp", "B1"],
        ["Y1"],
        kernel_shape=[k, k],
        pads=pads,
        strides=strides,
        name="QLinearConv1",
    )
    conv2 = helper.make_node(
        "QLinearConv",
        ["X", "x_scale", "x_zp", "W2", "w2_scale", "w2_zp", "y2_scale", "y2_zp", "B2"],
        ["Y2"],
        kernel_shape=[k, k],
        pads=pads,
        strides=strides,
        name="QLinearConv2",
    )
    concat = helper.make_node(
        "QLinearConcat",
        [
            "y_scale",
            "y_zp",
            "Y1",
            "y1_scale",
            "y1_zp",
            "Y2",
            "y2_scale",
            "y2_zp",
        ],
        ["Y"],
        axis=1,
        name="QLinearConcat1",
        domain="com.microsoft",
    )

    graph = helper.make_graph(
        [conv1, conv2, concat],
        "qconcat_two_branch",
        [X],
        [Y],
        initializers,
        value_info=[Y1, Y2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    model.ir_version = 9
    onnx.checker.check_model(model)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(OUT))
    print(f"Wrote {OUT} output_shape={[1, out_c, *spatial]}")


if __name__ == "__main__":
    main()
