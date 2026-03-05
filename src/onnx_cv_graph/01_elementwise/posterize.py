"""ポスタリゼーションの ONNX グラフ定義.

色数を減らす量子化処理.
output = Floor(input * levels) / levels
levels パラメータで段階数を制御 (2=2値化相当, 256=元画像相当).
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class PosterizeOp(OnnxGraphOp):
    """ポスタリゼーション.

    Mul(levels) → Floor → Div(levels) → Clip(0,1).
    """

    @property
    def op_name(self) -> str:
        return "posterize"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("levels", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"levels": (2.0, 32.0, 4.0)}

    def build_graph(self) -> GraphProto:
        zero = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="zero")
        one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")

        nodes = [
            helper.make_node("Mul", ["input", "levels"], ["scaled"]),
            helper.make_node("Floor", ["scaled"], ["floored"]),
            helper.make_node("Div", ["floored", "levels"], ["quantized"]),
            helper.make_node("Clip", ["quantized", "zero", "one"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        levels_vi = helper.make_tensor_value_info("levels", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi, levels_vi], [output_vi],
            initializer=[zero, one],
        )
