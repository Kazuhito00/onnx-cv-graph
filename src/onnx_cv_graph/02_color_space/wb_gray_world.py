"""Gray World ホワイトバランスの ONNX グラフ定義.

各チャネルの平均が全チャネル平均の平均と等しくなるようゲインを算出する.
  ch_mean = ReduceMean(input, axes=[0,2,3])  → (3,)
  global_mean = Mean(ch_mean)                → scalar
  gain = global_mean / (ch_mean + eps)       → (3,)
  output = input * gain → Clip(0,1)
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class WbGrayWorldOp(OnnxGraphOp):
    """Gray World ホワイトバランス (自動補正, パラメータなし).

    ReduceMean → Div → Mul → Clip(0,1).
    """

    @property
    def op_name(self) -> str:
        return "wb_gray_world"

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
            # 各チャネルの空間平均: (N, 3, H, W) → (N, 3, 1, 1)
            helper.make_node("ReduceMean", ["input"], ["ch_mean"], axes=[2, 3], keepdims=1),

            # 全チャネル平均: (N, 3, 1, 1) → (N, 1, 1, 1)
            helper.make_node("ReduceMean", ["ch_mean"], ["global_mean"], axes=[1], keepdims=1),

            # ゲイン: global_mean / (ch_mean + eps)
            helper.make_node("Add", ["ch_mean", "eps"], ["ch_mean_safe"]),
            helper.make_node("Div", ["global_mean", "ch_mean_safe"], ["gain"]),

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
