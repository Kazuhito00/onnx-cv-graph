"""チャネル別オートレベル補正 (Auto Levels) の ONNX グラフ定義.

Photoshop の「自動レベル補正」相当。R/G/B チャネルそれぞれ独立に
Min-Max 正規化を行い、各チャネルのダイナミックレンジを最大化する。
色かぶり補正の効果がある。
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class AutoLevelsOp(OnnxGraphOp):
    """チャネル別オートレベル補正.

    入力 (N,3,H,W) float32 の各チャネルについて
    ピクセル値の最小値・最大値を求め [0, 1] に線形スケーリングする。
    MinMaxNormOp の軸違いバージョン (axes=[2,3] でチャネル独立)。
    """

    @property
    def op_name(self) -> str:
        return "auto_levels"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        # H, W 軸で min/max を計算 → (N,3,1,1) チャネル独立
        reduce_min = helper.make_node(
            "ReduceMin", ["input"], ["ch_min"], axes=[2, 3], keepdims=1,
        )
        reduce_max = helper.make_node(
            "ReduceMax", ["input"], ["ch_max"], axes=[2, 3], keepdims=1,
        )

        # range = max - min
        sub_range = helper.make_node("Sub", ["ch_max", "ch_min"], ["ch_range"])

        # ゼロ除算回避: range が 0 の場合に eps を使用
        eps = numpy_helper.from_array(np.array(1e-8, dtype=np.float32), "eps")
        safe_range = helper.make_node("Max", ["ch_range", "eps"], ["safe_range"])

        # (input - ch_min) / safe_range
        sub_node = helper.make_node("Sub", ["input", "ch_min"], ["shifted"])
        div_node = helper.make_node("Div", ["shifted", "safe_range"], ["normalized"])

        # 画像ドメイン保証: [0, 1] にクリップ
        clip_min = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "clip_min")
        clip_max = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "clip_max")
        clip_node = helper.make_node("Clip", ["normalized", "clip_min", "clip_max"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [reduce_min, reduce_max, sub_range, safe_range, sub_node, div_node, clip_node],
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=[eps, clip_min, clip_max],
        )
