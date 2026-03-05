"""チャネル抽出の ONNX グラフ定義.

指定チャネル (R/G/B) を取り出し、3ch に複製して出力する.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class ChannelExtractOp(OnnxGraphOp):
    """チャネル抽出.

    指定チャネルを Gather(axis=1) で取り出し、Expand で 3ch に複製する.
    """

    def __init__(self, channel: str = "r"):
        self._channel = channel
        self._index = {"r": 0, "g": 1, "b": 2}[channel]

    @property
    def op_name(self) -> str:
        return f"channel_{self._channel}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        return [cls("r"), cls("g"), cls("b")]

    def build_graph(self) -> GraphProto:
        idx = np.array([self._index], dtype=np.int64)
        idx_init = numpy_helper.from_array(idx, name="ch_index")

        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        nodes = [
            helper.make_node("Gather", ["input", "ch_index"], ["single_ch"], axis=1),
            helper.make_node("Expand", ["single_ch", "expand_shape"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes,
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=[idx_init, expand_init],
        )
