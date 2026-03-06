"""uint8→float32 変換の ONNX グラフ定義.

[0,255] uint8 画像を [0,1] float32 に変換する.
Cast(float32) → Div(255).
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class Uint8ToFloatOp(OnnxGraphOp):
    """uint8 [0,255] → float32 [0,1] 変換."""

    @property
    def op_name(self) -> str:
        return "uint8_to_float"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.UINT8, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]


    def build_graph(self) -> GraphProto:
        scale = numpy_helper.from_array(
            np.array([255.0], dtype=np.float32), name="scale",
        )

        nodes = [
            helper.make_node("Cast", ["input"], ["float_input"], to=TensorProto.FLOAT),
            helper.make_node("Div", ["float_input", "scale"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.UINT8, ["N", 3, "H", "W"],
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H", "W"],
        )

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
            initializer=[scale],
        )
