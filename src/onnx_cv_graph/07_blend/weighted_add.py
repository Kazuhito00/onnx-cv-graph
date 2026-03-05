"""加重加算の ONNX グラフ定義.

2枚の画像を alpha * input + beta * input2 + gamma で合成する.
OpenCV の addWeighted 相当.
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class WeightedAddOp(OnnxGraphOp):
    """加重加算.

    output = clip(alpha * input + beta * input2 + gamma, 0, 1)
    """

    @property
    def op_name(self) -> str:
        return "weighted_add"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("input2", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("alpha", TensorProto.FLOAT, [1]),
            ("beta", TensorProto.FLOAT, [1]),
            ("gamma", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {
            "alpha": (0.0, 2.0, 1.0),
            "beta": (0.0, 2.0, 1.0),
            "gamma": (-1.0, 1.0, 0.0),
        }

    def build_graph(self) -> GraphProto:
        # alpha * input
        mul1 = helper.make_node("Mul", ["input", "alpha"], ["a_img"])
        # beta * input2
        mul2 = helper.make_node("Mul", ["input2", "beta"], ["b_img"])
        # a_img + b_img
        add1 = helper.make_node("Add", ["a_img", "b_img"], ["ab"])
        # ab + gamma
        add2 = helper.make_node("Add", ["ab", "gamma"], ["raw"])
        # Clip(0, 1)
        clip_min = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "clip_min")
        clip_max = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "clip_max")
        clip_node = helper.make_node("Clip", ["raw", "clip_min", "clip_max"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        input2_vi = helper.make_tensor_value_info("input2", TensorProto.FLOAT, ["N", 3, "H", "W"])
        alpha_vi = helper.make_tensor_value_info("alpha", TensorProto.FLOAT, [1])
        beta_vi = helper.make_tensor_value_info("beta", TensorProto.FLOAT, [1])
        gamma_vi = helper.make_tensor_value_info("gamma", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [mul1, mul2, add1, add2, clip_node],
            self.op_name,
            [input_vi, input2_vi, alpha_vi, beta_vi, gamma_vi],
            [output_vi],
            initializer=[clip_min, clip_max],
        )
