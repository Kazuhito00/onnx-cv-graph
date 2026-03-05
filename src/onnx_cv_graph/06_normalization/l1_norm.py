"""L1 正規化の ONNX グラフ定義.

各画像のピクセル値を L1 ノルムで除算して正規化する.
出力は [0, 1] 範囲に収まるよう Clip する.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class L1NormOp(OnnxGraphOp):
    """L1 正規化.

    入力 (N,3,H,W) float32 の各画像について
    x / (||x||_1 + eps) を計算する.
    出力は Clip(0, 1) で値域を保証する.
    """

    @property
    def op_name(self) -> str:
        return "l1_norm"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        # C, H, W 軸で L1 ノルムを計算 → (N,1,1,1)
        axes = numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64), "axes")

        # |x|
        abs_node = helper.make_node("Abs", ["input"], ["x_abs"])
        # sum(|x|) → (N,1,1,1)
        reduce_sum = helper.make_node(
            "ReduceSum", ["x_abs", "axes"], ["l1_norm"], keepdims=1,
        )

        # ゼロ除算回避
        eps = numpy_helper.from_array(np.array(1e-8, dtype=np.float32), "eps")
        add_eps = helper.make_node("Add", ["l1_norm", "eps"], ["safe_norm"])

        # x / norm
        div_node = helper.make_node("Div", ["input", "safe_norm"], ["normalized"])

        # Clip(0, 1)
        clip_min = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "clip_min")
        clip_max = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "clip_max")
        clip_node = helper.make_node("Clip", ["normalized", "clip_min", "clip_max"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [abs_node, reduce_sum, add_eps, div_node, clip_node],
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=[axes, eps, clip_min, clip_max],
        )
