"""float32→uint8 量子化の ONNX グラフ定義.

[0,1] float32 画像を [0,255] uint8 に変換する.
Mul(255) → Clip(0, 255) → Round → Cast(uint8).
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class FloatToUint8Op(OnnxGraphOp):
    """float32 [0,1] → uint8 [0,255] 量子化."""

    @property
    def op_name(self) -> str:
        return "float_to_uint8"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.UINT8, ["N", 3, "H", "W"])]


    def build_graph(self) -> GraphProto:
        scale = numpy_helper.from_array(
            np.array([255.0], dtype=np.float32), name="scale",
        )
        zero = numpy_helper.from_array(
            np.array([0.0], dtype=np.float32), name="zero",
        )
        max_val = numpy_helper.from_array(
            np.array([255.0], dtype=np.float32), name="max_val",
        )

        nodes = [
            helper.make_node("Mul", ["input", "scale"], ["scaled"]),
            helper.make_node("Clip", ["scaled", "zero", "max_val"], ["clipped"]),
            helper.make_node("Round", ["clipped"], ["rounded"]),
            helper.make_node("Cast", ["rounded"], ["output"], to=TensorProto.UINT8),
        ]

        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", 3, "H", "W"],
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.UINT8, ["N", 3, "H", "W"],
        )

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
            initializer=[scale, zero, max_val],
        )
