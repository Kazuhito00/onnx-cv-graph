"""オーバーレイ合成の ONNX グラフ定義.

Photoshop のオーバーレイモード相当.
base < 0.5: output = 2 * base * blend
base >= 0.5: output = 1 - 2 * (1 - base) * (1 - blend)
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class OverlayOp(OnnxGraphOp):
    """オーバーレイ合成.

    Where で条件分岐:
    base < 0.5 → 2 * base * blend
    base >= 0.5 → 1 - 2 * (1 - base) * (1 - blend)
    """

    @property
    def op_name(self) -> str:
        return "overlay"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("input2", TensorProto.FLOAT, ["N", 3, "H", "W"]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {}

    def build_graph(self) -> GraphProto:
        two = numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two")
        one = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one")
        half = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half")
        clip_min = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "clip_min")
        clip_max = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "clip_max")

        # 暗部: 2 * base * blend
        mul_dark1 = helper.make_node("Mul", ["input", "input2"], ["base_x_blend"])
        mul_dark2 = helper.make_node("Mul", ["base_x_blend", "two"], ["dark"])

        # 明部: 1 - 2 * (1 - base) * (1 - blend)
        sub_base = helper.make_node("Sub", ["one", "input"], ["inv_base"])
        sub_blend = helper.make_node("Sub", ["one", "input2"], ["inv_blend"])
        mul_light1 = helper.make_node("Mul", ["inv_base", "inv_blend"], ["inv_prod"])
        mul_light2 = helper.make_node("Mul", ["inv_prod", "two"], ["inv_prod2"])
        sub_light = helper.make_node("Sub", ["one", "inv_prod2"], ["light"])

        # 条件: base < 0.5
        cond = helper.make_node("Less", ["input", "half"], ["cond"])

        # Where(cond, dark, light)
        where = helper.make_node("Where", ["cond", "dark", "light"], ["raw"])

        # Clip(0, 1)
        clip_node = helper.make_node("Clip", ["raw", "clip_min", "clip_max"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        input2_vi = helper.make_tensor_value_info("input2", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [mul_dark1, mul_dark2, sub_base, sub_blend,
             mul_light1, mul_light2, sub_light, cond, where, clip_node],
            self.op_name,
            [input_vi, input2_vi],
            [output_vi],
            initializer=[two, one, half, clip_min, clip_max],
        )
