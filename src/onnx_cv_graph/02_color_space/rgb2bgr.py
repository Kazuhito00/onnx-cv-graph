"""RGB→BGR チャネルスワップの ONNX グラフ定義.

Gather(axis=1, indices=[2,1,0]) でチャネル順を反転する.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class Rgb2BgrOp(OnnxGraphOp):
    """RGB→BGR チャネルスワップ.

    入力 (N,3,H,W) の RGB チャネルを BGR に並べ替える.
    """

    @property
    def op_name(self) -> str:
        return "rgb2bgr"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        indices = np.array([2, 1, 0], dtype=np.int64)
        indices_init = numpy_helper.from_array(indices, name="bgr_indices")

        gather_node = helper.make_node("Gather", ["input", "bgr_indices"], ["output"], axis=1)

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [gather_node],
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=[indices_init],
        )
