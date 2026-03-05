"""ソラリゼーションの ONNX グラフ定義.

threshold 以上の画素を反転する.
output = Where(input >= threshold, 1.0 - input, input)
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class SolarizeOp(OnnxGraphOp):
    """ソラリゼーション.

    GreaterOrEqual → Where(1-input, input).
    """

    @property
    def op_name(self) -> str:
        return "solarize"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("threshold", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"threshold": (0.0, 1.0, 0.5)}

    def build_graph(self) -> GraphProto:
        one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")

        nodes = [
            helper.make_node("GreaterOrEqual", ["input", "threshold"], ["mask"]),
            helper.make_node("Sub", ["one", "input"], ["inverted"]),
            helper.make_node("Where", ["mask", "inverted", "input"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        thr_vi = helper.make_tensor_value_info("threshold", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi, thr_vi], [output_vi],
            initializer=[one],
        )
