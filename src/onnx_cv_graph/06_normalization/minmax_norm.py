"""Min-Max 正規化の ONNX グラフ定義.

各画像のピクセル値を [0, 1] 範囲に線形スケーリングする.
(x - min) / (max - min)
均一画像 (max == min) の場合はゼロ除算を避けるため 0.0 を出力する.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class MinMaxNormOp(OnnxGraphOp):
    """Min-Max 正規化.

    入力 (N,3,H,W) float32 の各画像について
    ピクセル値の最小値・最大値を求め [0, 1] に線形スケーリングする.
    """

    @property
    def op_name(self) -> str:
        return "minmax_norm"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        # C, H, W 軸で min/max を計算 → (N,1,1,1)
        # opset 17 では ReduceMin/ReduceMax は axes を属性で指定する
        reduce_min = helper.make_node(
            "ReduceMin", ["input"], ["x_min"], axes=[1, 2, 3], keepdims=1,
        )
        reduce_max = helper.make_node(
            "ReduceMax", ["input"], ["x_max"], axes=[1, 2, 3], keepdims=1,
        )

        # range = max - min
        sub_range = helper.make_node("Sub", ["x_max", "x_min"], ["x_range"])

        # ゼロ除算回避: range が 0 の場合に eps を使用
        eps = numpy_helper.from_array(np.array(1e-8, dtype=np.float32), "eps")
        safe_range = helper.make_node("Max", ["x_range", "eps"], ["safe_range"])

        # (input - min) / safe_range
        sub_node = helper.make_node("Sub", ["input", "x_min"], ["shifted"])
        div_node = helper.make_node("Div", ["shifted", "safe_range"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [reduce_min, reduce_max, sub_range, safe_range, sub_node, div_node],
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=[eps],
        )
