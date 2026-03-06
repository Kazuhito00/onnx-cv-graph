"""ピクセル平均減算 (Caffe/detectron2 系) の ONNX グラフ定義.

入力 [0,1] を 255 倍してから RGB チャネルごとの平均 [123.675, 116.28, 103.53] を減算する.
detectron2, mmdetection 等で使われる標準的な前処理. 画像ドメイン → ML ドメイン.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class PixelMeanSubOp(OnnxGraphOp):
    """ピクセル平均減算 (Caffe/detectron2 系).

    output = input * 255 - [123.675, 116.28, 103.53]
    """

    @property
    def op_name(self) -> str:
        return "pixel_mean_sub"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]


    def build_graph(self) -> GraphProto:
        scale = np.array([255.0], dtype=np.float32)
        scale_init = numpy_helper.from_array(scale, name="scale")

        # RGB 順の pixel mean (1, 3, 1, 1) でブロードキャスト
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 3, 1, 1)
        mean_init = numpy_helper.from_array(mean, name="pixel_mean")

        mul_node = helper.make_node("Mul", ["input", "scale"], ["scaled"])
        sub_node = helper.make_node("Sub", ["scaled", "pixel_mean"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [mul_node, sub_node],
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=[scale_init, mean_init],
        )
