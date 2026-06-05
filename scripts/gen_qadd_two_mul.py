#!/usr/bin/env python3
"""Generate qadd_two_mul.onnx: shared input, two QLinearMul, one QLinearAdd (F-a)."""
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
    / "qadd_two_mul.onnx"
)


def scalar(name: str, val, dtype):
    return helper.make_tensor(name, dtype, [], [val])


def main() -> None:
    shape = (1, 16, 1, 16)
    initializers = [
        scalar("c2_scale", 1.0, TensorProto.FLOAT),
        scalar("c2_zp", 0, TensorProto.INT8),
        helper.make_tensor("const2", TensorProto.INT8, [], np.array([2], dtype=np.int8).tobytes(), raw=True),
        scalar("c3_scale", 1.0, TensorProto.FLOAT),
        scalar("c3_zp", 0, TensorProto.INT8),
        helper.make_tensor("const3", TensorProto.INT8, [], np.array([3], dtype=np.int8).tobytes(), raw=True),
        scalar("y_scale", 1.0, TensorProto.FLOAT),
        scalar("y_zp", 0, TensorProto.INT8),
        scalar("x_scale", 1.0, TensorProto.FLOAT),
        scalar("x_zp", 0, TensorProto.INT8),
    ]
    X = helper.make_tensor_value_info("X", TensorProto.INT8, list(shape))
    Y1 = "Y_mul2"
    Y2 = "Y_mul3"
    Y = "Y"
    mul2 = helper.make_node(
        "QLinearMul",
        ["const2", "c2_scale", "c2_zp", "X", "x_scale", "x_zp", "y_scale", "y_zp"],
        [Y1],
        name="Mul2",
        domain="com.microsoft",
    )
    mul3 = helper.make_node(
        "QLinearMul",
        ["const3", "c3_scale", "c3_zp", "X", "x_scale", "x_zp", "y_scale", "y_zp"],
        [Y2],
        name="Mul3",
        domain="com.microsoft",
    )
    add = helper.make_node(
        "QLinearAdd",
        [Y1, "y_scale", "y_zp", Y2, "y_scale", "y_zp", "y_scale", "y_zp"],
        [Y],
        name="QLinearAdd1",
        domain="com.microsoft",
    )
    Y_info = helper.make_tensor_value_info(Y, TensorProto.INT8, list(shape))
    graph = helper.make_graph([mul2, mul3, add], "qadd_two_mul", [X], [Y_info], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    model.ir_version = 9
    onnx.checker.check_model(model)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
