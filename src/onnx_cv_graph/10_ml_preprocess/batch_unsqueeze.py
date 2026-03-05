"""バッチ次元追加の ONNX グラフ定義.

単一画像 (C, H, W) にバッチ次元を追加して (1, C, H, W) にする.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class BatchUnsqueezeOp(OnnxGraphOp):
    """バッチ次元追加 = Unsqueeze(axis=0)."""

    @property
    def op_name(self) -> str:
        return "batch_unsqueeze"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, [3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, [1, 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        axes = numpy_helper.from_array(
            np.array([0], dtype=np.int64), name="axes",
        )

        node = helper.make_node("Unsqueeze", ["input", "axes"], ["output"])

        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [3, "H", "W"],
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 3, "H", "W"],
        )

        return helper.make_graph(
            [node], self.op_name, [input_vi], [output_vi],
            initializer=[axes],
        )
