"""切り詰め閾値処理の ONNX グラフ定義.

gray > threshold → threshold, それ以外 → gray.
OpenCV の THRESH_TRUNC 相当.
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class ThreshTruncOp(OnnxGraphOp):
    """切り詰め閾値処理.

    gray > threshold の画素を threshold で切り詰める.
    ノード構成: Mul → ReduceSum → Min → Expand.
    """

    @property
    def op_name(self) -> str:
        return "thresh_trunc"

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

        # グレースケール化
        mul_node = helper.make_node("Mul", ["input", "luma_weights"], ["weighted"])
        reduce_node = helper.make_node("ReduceSum", ["weighted", "axes"], ["gray"], keepdims=1)

        # Min(gray, threshold) で切り詰め
        min_node = helper.make_node("Min", ["gray", "threshold"], ["trunc_1ch"])

        # 3ch 拡張
        expand_node = helper.make_node("Expand", ["trunc_1ch", "expand_shape"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        threshold_vi = helper.make_tensor_value_info("threshold", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [mul_node, reduce_node, min_node, expand_node],
            self.op_name,
            [input_vi, threshold_vi],
            [output_vi],
            initializer=[weights_init, axes_init, expand_shape_init],
        )
