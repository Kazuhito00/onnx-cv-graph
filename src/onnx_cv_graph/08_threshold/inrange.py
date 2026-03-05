"""範囲内抽出の ONNX グラフ定義.

lower ≤ gray ≤ upper の画素を 1.0、それ以外を 0.0 にする.
OpenCV の inRange 相当 (グレースケール版).
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class InrangeOp(OnnxGraphOp):
    """範囲内抽出.

    lower ≤ gray ≤ upper → 1.0, それ以外 → 0.0.
    ノード構成: Mul → ReduceSum → GreaterOrEqual → LessOrEqual → And → Cast → Expand.
    """

    @property
    def op_name(self) -> str:
        return "inrange"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("lower", TensorProto.FLOAT, [1]),
            ("upper", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {
            "lower": (0.0, 1.0, 0.2),
            "upper": (0.0, 1.0, 0.8),
        }

    def build_graph(self) -> GraphProto:
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        weights_init = numpy_helper.from_array(weights, name="luma_weights")
        axes = np.array([1], dtype=np.int64)
        axes_init = numpy_helper.from_array(axes, name="axes")
        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_shape_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        # グレースケール化
        mul_node = helper.make_node("Mul", ["input", "luma_weights"], ["weighted"])
        reduce_node = helper.make_node("ReduceSum", ["weighted", "axes"], ["gray"], keepdims=1)

        # gray >= lower
        ge_node = helper.make_node("GreaterOrEqual", ["gray", "lower"], ["ge_mask"])
        # gray <= upper
        le_node = helper.make_node("LessOrEqual", ["gray", "upper"], ["le_mask"])
        # AND: 両方満たす画素
        and_node = helper.make_node("And", ["ge_mask", "le_mask"], ["in_range"])

        # bool → float
        cast_node = helper.make_node("Cast", ["in_range"], ["range_1ch"], to=TensorProto.FLOAT)

        # 3ch 拡張
        expand_node = helper.make_node("Expand", ["range_1ch", "expand_shape"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        lower_vi = helper.make_tensor_value_info("lower", TensorProto.FLOAT, [1])
        upper_vi = helper.make_tensor_value_info("upper", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [mul_node, reduce_node, ge_node, le_node, and_node, cast_node, expand_node],
            self.op_name,
            [input_vi, lower_vi, upper_vi],
            [output_vi],
            initializer=[weights_init, axes_init, expand_shape_init],
        )
