"""色温度調整の ONNX グラフ定義.

temperature パラメータで R/B チャネルのゲインを調整する.
temperature > 0 で暖色 (R↑, B↓)、< 0 で寒色 (R↓, B↑).

計算式:
  gain_r = 1 + temperature
  gain_b = 1 - temperature
  output = input * [gain_r, 1, gain_b] → Clip(0, 1)
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class ColorTemperatureOp(OnnxGraphOp):
    """色温度調整.

    チャネルごとゲイン [1+t, 1, 1-t] を乗算して Clip(0,1) する.
    """

    @property
    def op_name(self) -> str:
        return "color_temperature"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("temperature", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"temperature": (-0.5, 0.5, 0.0)}

    def build_graph(self) -> GraphProto:
        one = np.array([1.0], dtype=np.float32)
        one_init = numpy_helper.from_array(one, name="one")
        zero = np.array([0.0], dtype=np.float32)
        zero_init = numpy_helper.from_array(zero, name="zero")

        # ゲインベクトル組み立て用の shape (1, 3, 1, 1)
        # gain = [1+t, 1, 1-t] を動的に構築
        unsqueeze_axes = np.array([0, 2, 3], dtype=np.int64)
        unsqueeze_axes_init = numpy_helper.from_array(unsqueeze_axes, name="unsqueeze_axes")

        initializers = [one_init, zero_init, unsqueeze_axes_init]

        nodes = []

        # gain_r = 1 + temperature
        nodes.append(helper.make_node("Add", ["one", "temperature"], ["gain_r"]))
        # gain_b = 1 - temperature
        nodes.append(helper.make_node("Sub", ["one", "temperature"], ["gain_b"]))

        # [gain_r, 1, gain_b] → (3,) を Concat で構築
        nodes.append(helper.make_node("Concat", ["gain_r", "one", "gain_b"], ["gain_flat"], axis=0))

        # (3,) → (1, 3, 1, 1) に Unsqueeze
        nodes.append(helper.make_node("Unsqueeze", ["gain_flat", "unsqueeze_axes"], ["gain"]))

        # input * gain → Clip(0, 1)
        nodes.append(helper.make_node("Mul", ["input", "gain"], ["scaled"]))
        nodes.append(helper.make_node("Clip", ["scaled", "zero", "one"], ["output"]))

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        temp_vi = helper.make_tensor_value_info("temperature", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes,
            self.op_name,
            [input_vi, temp_vi],
            [output_vi],
            initializer=initializers,
        )
