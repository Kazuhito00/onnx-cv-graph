"""チャネルごと平均減算の ONNX グラフ定義.

Caffe 系モデルの前処理で使われる. 入力画像から各チャネルの平均値を減算する.
画像ドメイン → ML ドメイン. mean は固定値 (VGG-Face の標準値).
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class ChannelMeanSubOp(OnnxGraphOp):
    """チャネルごと平均減算.

    入力 [0,1] (RGB) から各チャネルの平均を減算する.
    デフォルト mean は [0.485, 0.456, 0.406] (ImageNet 平均を [0,1] スケールで).
    """

    @property
    def op_name(self) -> str:
        return "channel_mean_sub"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_domain(self) -> str:
        return "ml"

    def build_graph(self) -> GraphProto:
        # デフォルト: ImageNet 平均 ([0,1] スケール)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        mean_init = numpy_helper.from_array(mean, name="mean")

        sub_node = helper.make_node("Sub", ["input", "mean"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [sub_node],
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=[mean_init],
        )
