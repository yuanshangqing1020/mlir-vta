#!/usr/bin/env python3
"""Generate lenet5_mini.onnx: 2× (QLinearConv + Relu) chain on 5×5 (increment G).

Uses the same spatial size as conv_relu_debug so fsim_nn numeric golden aligns
with NumPy QLinearConv + int8 ReLU reference (16×16 VTA blocks differ from ORT).
"""
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
    / "lenet5_mini.onnx"
)

IN_C, OUT_C = 3, 3
H, W = 5, 5
kernel = [3, 3]
pads = [1, 1, 1, 1]
strides = [1, 1]
shape = [1, IN_C, H, W]


def scalar(name, val, dtype):
    return helper.make_tensor(name, dtype, [], [val])


def main() -> None:
    np.random.seed(42)
    w_shape = (OUT_C, IN_C, 3, 3)
    w1 = np.random.randint(-128, 127, size=w_shape, dtype=np.int8)
    w2 = np.random.randint(-128, 127, size=w_shape, dtype=np.int8)
    b1 = np.random.randint(-500, 500, size=(OUT_C,), dtype=np.int32)
    b2 = np.random.randint(-500, 500, size=(OUT_C,), dtype=np.int32)

    initializers = [
        scalar("in_scale", 0.0315, TensorProto.FLOAT),
        scalar("in_zp", -128, TensorProto.INT8),
        scalar("w1_scale", 0.0050, TensorProto.FLOAT),
        scalar("w1_zp", 0, TensorProto.INT8),
        scalar("mid_scale", 0.0241, TensorProto.FLOAT),
        scalar("mid_zp", -128, TensorProto.INT8),
        scalar("w2_scale", 0.0048, TensorProto.FLOAT),
        scalar("w2_zp", 0, TensorProto.INT8),
        scalar("out_scale", 0.0241, TensorProto.FLOAT),
        scalar("out_zp", -128, TensorProto.INT8),
        helper.make_tensor("W1", TensorProto.INT8, w_shape, w1.tobytes(), raw=True),
        helper.make_tensor("B1", TensorProto.INT32, [OUT_C], b1.tobytes(), raw=True),
        helper.make_tensor("W2", TensorProto.INT8, w_shape, w2.tobytes(), raw=True),
        helper.make_tensor("B2", TensorProto.INT32, [OUT_C], b2.tobytes(), raw=True),
    ]

    X = helper.make_tensor_value_info("X", TensorProto.INT8, shape)
    Y_conv1 = helper.make_tensor_value_info("Y_conv1", TensorProto.INT8, shape)
    Y_relu1 = helper.make_tensor_value_info("Y_relu1", TensorProto.INT8, shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.INT8, shape)

    conv1 = helper.make_node(
        "QLinearConv",
        ["X", "in_scale", "in_zp", "W1", "w1_scale", "w1_zp", "mid_scale", "mid_zp", "B1"],
        ["Y_conv1"],
        kernel_shape=kernel,
        pads=pads,
        strides=strides,
        name="QLinearConv1",
    )
    relu1 = helper.make_node("Relu", inputs=["Y_conv1"], outputs=["Y_relu1"], name="Relu1")
    conv2 = helper.make_node(
        "QLinearConv",
        ["Y_relu1", "mid_scale", "mid_zp", "W2", "w2_scale", "w2_zp", "out_scale", "out_zp", "B2"],
        ["Y_conv2"],
        kernel_shape=kernel,
        pads=pads,
        strides=strides,
        name="QLinearConv2",
    )
    relu2 = helper.make_node("Relu", inputs=["Y_conv2"], outputs=["Y"], name="Relu2")

    graph = helper.make_graph(
        [conv1, relu1, conv2, relu2],
        "lenet5_mini",
        [X],
        [Y],
        initializers,
        value_info=[Y_conv1, Y_relu1],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(OUT))
    print(f"Wrote {OUT} shape={shape}")


if __name__ == "__main__":
    main()
