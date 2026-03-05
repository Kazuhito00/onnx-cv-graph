"""露出調整の ONNX グラフ定義.

output = input * exposure → Clip(0,1)
exposure > 1 で明るく (露出オーバー), < 1 で暗く (露出アンダー).
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class ExposureOp(OnnxGraphOp):
    """露出調整.

    Mul(exposure) → Clip(0,1).
    """

    @property
    def op_name(self) -> str:
        return "exposure"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("exposure", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"exposure": (0.0, 5.0, 1.0)}

    def build_graph(self) -> GraphProto:
        zero = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="zero")
        one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")

        nodes = [
            helper.make_node("Mul", ["input", "exposure"], ["scaled"]),
            helper.make_node("Clip", ["scaled", "zero", "one"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        exp_vi = helper.make_tensor_value_info("exposure", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi, exp_vi], [output_vi],
            initializer=[zero, one],
        )
