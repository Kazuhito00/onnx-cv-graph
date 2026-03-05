"""適応的閾値処理 (平均) の ONNX グラフ定義.

局所平均との差で2値化する. OpenCV の adaptiveThreshold(ADAPTIVE_THRESH_MEAN_C) 相当.
block_size ごとに別モデルを生成する (3/5/7).
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class AdaptiveThreshMeanOp(OnnxGraphOp):
    """適応的閾値処理 (平均).

    グレースケール化 → 局所平均 (AveragePool) → gray > (local_mean - C) で2値化.
    """

    def __init__(self, kernel_size: int = 3):
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        return f"adaptive_thresh_mean_{self._kernel_size}x{self._kernel_size}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("C", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"C": (-0.5, 0.5, 0.02)}

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        return [cls(3), cls(5), cls(7)]

    def build_graph(self) -> GraphProto:
        k = self._kernel_size
        pad = k // 2

        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        weights_init = numpy_helper.from_array(weights, name="luma_weights")
        axes = np.array([1], dtype=np.int64)
        axes_init = numpy_helper.from_array(axes, name="axes")
        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_shape_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        # グレースケール化
        mul_node = helper.make_node("Mul", ["input", "luma_weights"], ["weighted"])
        reduce_node = helper.make_node("ReduceSum", ["weighted", "axes"], ["gray"], keepdims=1)

        # 局所平均: AveragePool (count_include_pad でパディング部を含めない)
        avgpool_node = helper.make_node(
            "AveragePool", ["gray"], ["local_mean"],
            kernel_shape=[k, k],
            pads=[pad, pad, pad, pad],
            count_include_pad=0,
        )

        # local_mean - C → 適応的閾値
        sub_node = helper.make_node("Sub", ["local_mean", "C"], ["adaptive_thr"])

        # gray > adaptive_thr → bool → float
        greater_node = helper.make_node("Greater", ["gray", "adaptive_thr"], ["mask"])
        cast_node = helper.make_node("Cast", ["mask"], ["bin_1ch"], to=TensorProto.FLOAT)

        # 3ch 拡張
        expand_node = helper.make_node("Expand", ["bin_1ch", "expand_shape"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        c_vi = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [mul_node, reduce_node, avgpool_node, sub_node, greater_node, cast_node, expand_node],
            self.op_name,
            [input_vi, c_vi],
            [output_vi],
            initializer=[weights_init, axes_init, expand_shape_init],
        )
