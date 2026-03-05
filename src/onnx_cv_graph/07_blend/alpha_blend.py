"""アルファブレンドの ONNX グラフ定義.

2枚の画像を alpha で線形補間する.
output = alpha * input + (1 - alpha) * input2
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class AlphaBlendOp(OnnxGraphOp):
    """アルファブレンド.

    output = alpha * input + (1 - alpha) * input2
    """

    @property
    def op_name(self) -> str:
        return "alpha_blend"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("input2", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("alpha", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"alpha": (0.0, 1.0, 0.5)}

    def build_graph(self) -> GraphProto:
        one = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one")

        # alpha * input
        mul1 = helper.make_node("Mul", ["input", "alpha"], ["a_img"])
        # 1 - alpha
        sub1 = helper.make_node("Sub", ["one", "alpha"], ["one_minus_alpha"])
        # (1 - alpha) * input2
        mul2 = helper.make_node("Mul", ["input2", "one_minus_alpha"], ["b_img"])
        # a_img + b_img
        add1 = helper.make_node("Add", ["a_img", "b_img"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        input2_vi = helper.make_tensor_value_info("input2", TensorProto.FLOAT, ["N", 3, "H", "W"])
        alpha_vi = helper.make_tensor_value_info("alpha", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [mul1, sub1, mul2, add1],
            self.op_name,
            [input_vi, input2_vi, alpha_vi],
            [output_vi],
            initializer=[one],
        )
