"""White Patch ホワイトバランスの ONNX グラフ定義.

各チャネルの最大値が 1.0 になるようゲインを算出する.
  ch_max = ReduceMax(input, axes=[2,3])  → (N, 3, 1, 1)
  gain = 1.0 / (ch_max + eps)
  output = input * gain → Clip(0,1)
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class WbWhitePatchOp(OnnxGraphOp):
    """White Patch ホワイトバランス (自動補正, パラメータなし).

    ReduceMax → Div(1.0) → Mul → Clip(0,1).
    """

    @property
    def op_name(self) -> str:
        return "wb_white_patch"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        eps = numpy_helper.from_array(np.array([1e-7], dtype=np.float32), name="eps")
        zero = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="zero")
        one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")

        nodes = [
            # 各チャネルの空間最大値: (N, 3, H, W) → (N, 3, 1, 1)
            helper.make_node("ReduceMax", ["input"], ["ch_max"], axes=[2, 3], keepdims=1),

            # ゲイン: 1.0 / (ch_max + eps)
            helper.make_node("Add", ["ch_max", "eps"], ["ch_max_safe"]),
            helper.make_node("Div", ["one", "ch_max_safe"], ["gain"]),

            # input * gain → Clip(0, 1)
            helper.make_node("Mul", ["input", "gain"], ["scaled"]),
            helper.make_node("Clip", ["scaled", "zero", "one"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
            initializer=[eps, zero, one],
        )
