"""[0,1]→[0,255] スケーリングの ONNX グラフ定義.

uint8 レンジに変換する ML 前処理. 画像ドメイン → ML ドメイン.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class ScaleTo255Op(OnnxGraphOp):
    """[0,1]→[0,255] スケーリング.

    入力 [0,1] を 255 倍して [0,255] に変換する.
    """

    @property
    def op_name(self) -> str:
        return "scale_to_255"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]


    def build_graph(self) -> GraphProto:
        scale = np.array([255.0], dtype=np.float32)
        scale_init = numpy_helper.from_array(scale, name="scale")

        mul_node = helper.make_node("Mul", ["input", "scale"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [mul_node],
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=[scale_init],
        )
