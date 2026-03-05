"""彩度調整の ONNX グラフ定義.

グレースケール値と元画像を saturation パラメータで線形補間する.
output = gray * (1 - saturation) + input * saturation → Clip(0,1)
saturation=0 でグレースケール, =1 で元画像, >1 で彩度増幅.
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class SaturationOp(OnnxGraphOp):
    """彩度調整.

    グレースケール → Mul(1-s) + input*s → Clip(0,1).
    """

    @property
    def op_name(self) -> str:
        return "saturation"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("saturation", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"saturation": (0.0, 3.0, 1.0)}

    def build_graph(self) -> GraphProto:
        luma = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        luma_init = numpy_helper.from_array(luma, name="luma")
        axes = np.array([1], dtype=np.int64)
        axes_init = numpy_helper.from_array(axes, name="axes")
        one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")
        zero = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="zero")
        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        nodes = [
            # グレースケール: (N,3,H,W) → (N,1,H,W) → (N,3,H,W)
            helper.make_node("Mul", ["input", "luma"], ["weighted"]),
            helper.make_node("ReduceSum", ["weighted", "axes"], ["gray_1ch"], keepdims=1),
            helper.make_node("Expand", ["gray_1ch", "expand_shape"], ["gray"]),
            # (1 - saturation)
            helper.make_node("Sub", ["one", "saturation"], ["inv_sat"]),
            # gray * (1-s)
            helper.make_node("Mul", ["gray", "inv_sat"], ["gray_part"]),
            # input * s
            helper.make_node("Mul", ["input", "saturation"], ["color_part"]),
            # gray_part + color_part
            helper.make_node("Add", ["gray_part", "color_part"], ["blended"]),
            # Clip(0, 1)
            helper.make_node("Clip", ["blended", "zero", "one"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        sat_vi = helper.make_tensor_value_info("saturation", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi, sat_vi], [output_vi],
            initializer=[luma_init, axes_init, one, zero, expand_init],
        )
