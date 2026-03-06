"""[0,255]→[0,1] スケーリングの ONNX グラフ定義.

uint8 レンジの入力を [0,1] に正規化する. ML ドメイン → 画像ドメイン.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class ScaleFrom255Op(OnnxGraphOp):
    """[0,255]→[0,1] スケーリング.

    入力 [0,255] を 255 で割って [0,1] に変換する.
    """

    @property
    def op_name(self) -> str:
        return "scale_from_255"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]


    def build_graph(self) -> GraphProto:
        scale = np.array([255.0], dtype=np.float32)
        scale_init = numpy_helper.from_array(scale, name="scale")

        div_node = helper.make_node("Div", ["input", "scale"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [div_node],
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=[scale_init],
        )
