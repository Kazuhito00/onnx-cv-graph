"""バッチ次元除去の ONNX グラフ定義.

バッチ画像 (1, C, H, W) からバッチ次元を除去して (C, H, W) にする.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class BatchSqueezeOp(OnnxGraphOp):
    """バッチ次元除去 = Squeeze(axis=0)."""

    @property
    def op_name(self) -> str:
        return "batch_squeeze"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, [3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        axes = numpy_helper.from_array(
            np.array([0], dtype=np.int64), name="axes",
        )

        node = helper.make_node("Squeeze", ["input", "axes"], ["output"])

        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", 3, "H", "W"],
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [3, "H", "W"],
        )

        return helper.make_graph(
            [node], self.op_name, [input_vi], [output_vi],
            initializer=[axes],
        )
