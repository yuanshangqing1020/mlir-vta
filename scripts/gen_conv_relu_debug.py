#!/usr/bin/env python3
"""Generate QLinearConv -> Relu ONNX (conv_relu_debug.onnx) for increment E."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = Path(__file__).resolve().parents[2] / "standalone-vta" / "examples" / "onnx" / "conv_relu_debug.onnx"


def main() -> None:
    input_shape = (1, 3, 5, 5)
    out_channels = 3
    kernel_shape = (3, 3)
    pads = [1, 1, 1, 1]
    strides = [1, 1]
    N, in_c, H, W = input_shape
    kH, kW = kernel_shape
    out_H = (H + pads[0] + pads[2] - kH) // strides[0] + 1
    out_W = (W + pads[1] + pads[3] - kW) // strides[1] + 1
    output_shape = [N, out_channels, out_H, out_W]
    weight_shape = (out_channels, in_c, kH, kW)

    W_data = np.random.randint(-128, 127, size=weight_shape, dtype=np.int8)
    B_data = np.random.randint(-1000, 1000, size=(out_channels,), dtype=np.int32)

    def scalar(name, val, dtype):
        return helper.make_tensor(name, dtype, [], [val])

    initializers = [
        scalar("x_scale", 0.0315, TensorProto.FLOAT),
        scalar("x_zp", -128, TensorProto.INT8),
        helper.make_tensor("W", TensorProto.INT8, weight_shape, W_data.tobytes(), raw=True),
        scalar("w_scale", 0.0050, TensorProto.FLOAT),
        scalar("w_zp", 0, TensorProto.INT8),
        scalar("y_scale", 0.0241, TensorProto.FLOAT),
        scalar("y_zp", -128, TensorProto.INT8),
        helper.make_tensor("B", TensorProto.INT32, [out_channels], B_data.tobytes(), raw=True),
    ]

    X_info = helper.make_tensor_value_info("X", TensorProto.INT8, input_shape)
    conv_out = "Y_conv"
    relu_out = "Y"
    conv = helper.make_node(
        "QLinearConv",
        inputs=["X", "x_scale", "x_zp", "W", "w_scale", "w_zp", "y_scale", "y_zp", "B"],
        outputs=[conv_out],
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
    )
    relu = helper.make_node("Relu", inputs=[conv_out], outputs=[relu_out])
    Y_info = helper.make_tensor_value_info(relu_out, TensorProto.INT8, output_shape)
    graph = helper.make_graph([conv, relu], "conv_relu", [X_info], [Y_info], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(OUT))
    print(f"Wrote {OUT} output_shape={output_shape}")


if __name__ == "__main__":
    main()
