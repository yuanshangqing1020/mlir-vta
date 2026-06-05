#!/usr/bin/env python3
"""Generate a small two-layer QLinearConv ONNX for fsim_nn testing (16x16 spatial)."""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = Path(__file__).resolve().parents[2] / "standalone-vta" / "examples" / "onnx" / "two_qlinearconv_small.onnx"

channel = 16
dim = 16
kernel = [3, 3]
pads = [1, 1, 1, 1]
strides = [1, 1]


def scalar(name, val, dtype):
    return helper.make_tensor(name, dtype, [], [val])


def weight(name, shape):
    data = np.random.randint(-128, 127, size=shape, dtype=np.int8)
    return helper.make_tensor(name, TensorProto.INT8, shape, data.tobytes(), raw=True)


np.random.seed(42)
initializers = [
    scalar("in_scale", 0.025, TensorProto.FLOAT),
    scalar("in_zp", -128, TensorProto.INT8),
    scalar("w1_scale", 0.0166, TensorProto.FLOAT),
    scalar("w1_zp", 0, TensorProto.INT8),
    scalar("mid_scale", 0.0241, TensorProto.FLOAT),
    scalar("mid_zp", -128, TensorProto.INT8),
    scalar("w2_scale", 0.0089, TensorProto.FLOAT),
    scalar("w2_zp", 0, TensorProto.INT8),
    scalar("out_scale", 0.0256, TensorProto.FLOAT),
    scalar("out_zp", -128, TensorProto.INT8),
    weight("W1", (channel, channel, 3, 3)),
    helper.make_tensor("B1", TensorProto.INT32, [channel], np.full(channel, 10, dtype=np.int32).tobytes(), raw=True),
    weight("W2", (channel, channel, 3, 3)),
    helper.make_tensor("B2", TensorProto.INT32, [channel], np.full(channel, -5, dtype=np.int32).tobytes(), raw=True),
]

X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, channel, dim, dim])
Y1 = helper.make_tensor_value_info("Y1", TensorProto.INT8, [1, channel, dim, dim])
Y = helper.make_tensor_value_info("Y", TensorProto.INT8, [1, channel, dim, dim])

conv1 = helper.make_node(
    "QLinearConv",
    inputs=["X", "in_scale", "in_zp", "W1", "w1_scale", "w1_zp", "mid_scale", "mid_zp", "B1"],
    outputs=["Y1"],
    kernel_shape=kernel,
    pads=pads,
    strides=strides,
    name="QLinearConv1",
)
conv2 = helper.make_node(
    "QLinearConv",
    inputs=["Y1", "mid_scale", "mid_zp", "W2", "w2_scale", "w2_zp", "out_scale", "out_zp", "B2"],
    outputs=["Y"],
    kernel_shape=kernel,
    pads=pads,
    strides=strides,
    name="QLinearConv2",
)

graph = helper.make_graph([conv1, conv2], "two_qlinearconv_small", [X], [Y], initializers, value_info=[Y1])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.checker.check_model(model)
OUT.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, str(OUT))
print(f"Wrote {OUT}")
