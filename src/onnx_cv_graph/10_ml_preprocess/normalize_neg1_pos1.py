"""[-1,1] 正規化の ONNX グラフ定義.

GAN / CLIP 等のモデル入力用. [0,1] → [-1,1]. 画像ドメイン → ML ドメイン.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class NormalizeNeg1Pos1Op(OnnxGraphOp):
    """[-1,1] 正規化.

    入力 [0,1] を Mul(2) → Sub(1) で [-1,1] に変換する.
    """

    @property
    def op_name(self) -> str:
        return "normalize_neg1_pos1"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]


    def build_graph(self) -> GraphProto:
        two = np.array([2.0], dtype=np.float32)
        one = np.array([1.0], dtype=np.float32)
        two_init = numpy_helper.from_array(two, name="two")
        one_init = numpy_helper.from_array(one, name="one")

        mul_node = helper.make_node("Mul", ["input", "two"], ["scaled"])
        sub_node = helper.make_node("Sub", ["scaled", "one"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [mul_node, sub_node],
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=[two_init, one_init],
        )
