"""チャネル軸 L2 正規化の ONNX グラフ定義.

各ピクセルの RGB 3値を L2 ノルムで除算して正規化する.
正規化後の各ピクセルの L2 ノルムは 1 になる.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class L2NormChOp(OnnxGraphOp):
    """チャネル軸 L2 正規化.

    入力 (N,3,H,W) float32 の各ピクセルについて
    (R,G,B) / sqrt(R² + G² + B² + eps) を計算する.
    """

    @property
    def op_name(self) -> str:
        return "l2_norm_ch"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        # チャネル軸 (axis=1) で L2 ノルムを計算 → (N,1,H,W)
        axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")

        # x^2
        mul_node = helper.make_node("Mul", ["input", "input"], ["x_sq"])
        # sum(x^2, axis=1) → (N,1,H,W)
        reduce_sum = helper.make_node(
            "ReduceSum", ["x_sq", "axes"], ["sum_sq"], keepdims=1,
        )
        # sqrt
        sqrt_node = helper.make_node("Sqrt", ["sum_sq"], ["l2_norm"])

        # ゼロ除算回避
        eps = numpy_helper.from_array(np.array(1e-8, dtype=np.float32), "eps")
        add_eps = helper.make_node("Add", ["l2_norm", "eps"], ["safe_norm"])

        # x / norm
        div_node = helper.make_node("Div", ["input", "safe_norm"], ["normalized"])

        # Clip(0, 1)
        clip_min = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "clip_min")
        clip_max = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "clip_max")
        clip_node = helper.make_node("Clip", ["normalized", "clip_min", "clip_max"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [mul_node, reduce_sum, sqrt_node, add_eps, div_node, clip_node],
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=[axes, eps, clip_min, clip_max],
        )
