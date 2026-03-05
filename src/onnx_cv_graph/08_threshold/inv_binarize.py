"""逆2値化の ONNX グラフ定義.

gray > threshold → 0.0, それ以外 → 1.0 (通常の2値化の反転).
OpenCV の THRESH_BINARY_INV 相当.
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class InvBinarizeOp(OnnxGraphOp):
    """逆2値化.

    gray > threshold → 0.0, gray ≤ threshold → 1.0.
    ノード構成: Mul → ReduceSum → Greater → Cast → Sub → Expand.
    """

    @property
    def op_name(self) -> str:
        return "inv_binarize"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("threshold", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"threshold": (0.0, 1.0, 0.5)}

    def build_graph(self) -> GraphProto:
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        weights_init = numpy_helper.from_array(weights, name="luma_weights")
        axes = np.array([1], dtype=np.int64)
        axes_init = numpy_helper.from_array(axes, name="axes")
        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_shape_init = numpy_helper.from_array(expand_shape, name="expand_shape")
        one = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one")

        # グレースケール化
        mul_node = helper.make_node("Mul", ["input", "luma_weights"], ["weighted"])
        reduce_node = helper.make_node("ReduceSum", ["weighted", "axes"], ["gray"], keepdims=1)

        # gray > threshold → bool → float
        greater_node = helper.make_node("Greater", ["gray", "threshold"], ["mask"])
        cast_node = helper.make_node("Cast", ["mask"], ["bin_1ch"], to=TensorProto.FLOAT)

        # 反転: 1.0 - bin_1ch
        sub_node = helper.make_node("Sub", ["one", "bin_1ch"], ["inv_1ch"])

        # 3ch 拡張
        expand_node = helper.make_node("Expand", ["inv_1ch", "expand_shape"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        threshold_vi = helper.make_tensor_value_info("threshold", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [mul_node, reduce_node, greater_node, cast_node, sub_node, expand_node],
            self.op_name,
            [input_vi, threshold_vi],
            [output_vi],
            initializer=[weights_init, axes_init, expand_shape_init, one],
        )
