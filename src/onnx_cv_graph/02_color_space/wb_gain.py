"""チャネルゲイン式ホワイトバランスの ONNX グラフ定義.

R/G/B 各チャネルに独立したゲインを適用する.
output = input * [r_gain, g_gain, b_gain] → Clip(0,1)
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class WbGainOp(OnnxGraphOp):
    """チャネルゲイン式ホワイトバランス.

    Concat([r,g,b]) → Unsqueeze(1,3,1,1) → Mul → Clip(0,1).
    """

    @property
    def op_name(self) -> str:
        return "wb_gain"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("r_gain", TensorProto.FLOAT, [1]),
            ("g_gain", TensorProto.FLOAT, [1]),
            ("b_gain", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {
            "r_gain": (0.0, 3.0, 1.0),
            "g_gain": (0.0, 3.0, 1.0),
            "b_gain": (0.0, 3.0, 1.0),
        }

    def build_graph(self) -> GraphProto:
        zero = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="zero")
        one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")
        unsqueeze_axes = numpy_helper.from_array(
            np.array([0, 2, 3], dtype=np.int64), name="unsqueeze_axes",
        )

        nodes = [
            # [r_gain, g_gain, b_gain] → (3,)
            helper.make_node("Concat", ["r_gain", "g_gain", "b_gain"], ["gain_flat"], axis=0),
            # (3,) → (1, 3, 1, 1)
            helper.make_node("Unsqueeze", ["gain_flat", "unsqueeze_axes"], ["gain"]),
            # input * gain → Clip
            helper.make_node("Mul", ["input", "gain"], ["scaled"]),
            helper.make_node("Clip", ["scaled", "zero", "one"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        r_vi = helper.make_tensor_value_info("r_gain", TensorProto.FLOAT, [1])
        g_vi = helper.make_tensor_value_info("g_gain", TensorProto.FLOAT, [1])
        b_vi = helper.make_tensor_value_info("b_gain", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name,
            [input_vi, r_vi, g_vi, b_vi], [output_vi],
            initializer=[zero, one, unsqueeze_axes],
        )
