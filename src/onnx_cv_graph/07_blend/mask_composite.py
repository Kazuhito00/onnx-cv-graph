"""マスク合成の ONNX グラフ定義.

マスク画像を使って2枚の画像を合成する.
output = mask * input + (1 - mask) * input2
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class MaskCompositeOp(OnnxGraphOp):
    """マスク合成.

    output = mask * input + (1 - mask) * input2
    mask は [N, 1, H, W] の単チャネル画像 (0〜1).
    """

    @property
    def op_name(self) -> str:
        return "mask_composite"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("input2", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("mask", TensorProto.FLOAT, ["N", 1, "H", "W"]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {}

    def build_graph(self) -> GraphProto:
        one = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one")

        # mask * input
        mul1 = helper.make_node("Mul", ["input", "mask"], ["fg"])
        # 1 - mask
        sub1 = helper.make_node("Sub", ["one", "mask"], ["inv_mask"])
        # (1 - mask) * input2
        mul2 = helper.make_node("Mul", ["input2", "inv_mask"], ["bg"])
        # fg + bg
        add1 = helper.make_node("Add", ["fg", "bg"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        input2_vi = helper.make_tensor_value_info("input2", TensorProto.FLOAT, ["N", 3, "H", "W"])
        mask_vi = helper.make_tensor_value_info("mask", TensorProto.FLOAT, ["N", 1, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [mul1, sub1, mul2, add1],
            self.op_name,
            [input_vi, input2_vi, mask_vi],
            [output_vi],
            initializer=[one],
        )
