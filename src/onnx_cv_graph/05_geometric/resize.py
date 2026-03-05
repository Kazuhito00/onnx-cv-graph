"""リサイズの ONNX グラフ定義.

ONNX Resize オペレータで双線形補間リサイズを実現する.
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class ResizeOp(OnnxGraphOp):
    """リサイズ (双線形補間).

    入力 (N,3,H,W) float32 に対し、scale 倍のリサイズを適用する.
    """

    @property
    def op_name(self) -> str:
        return "resize"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("scale", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H_out", "W_out"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"scale": (0.1, 4.0, 1.0)}

    def build_graph(self) -> GraphProto:
        # scales = [1.0, 1.0, scale, scale]
        nc_scales = numpy_helper.from_array(
            np.array([1.0, 1.0], dtype=np.float32), "nc_scales"
        )
        concat_node = helper.make_node(
            "Concat", ["nc_scales", "scale", "scale"], ["scales"], axis=0
        )

        roi = numpy_helper.from_array(np.array([], dtype=np.float32), "roi")

        resize_node = helper.make_node(
            "Resize",
            inputs=["input", "roi", "scales"],
            outputs=["output"],
            mode="linear",
        )

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        scale_vi = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H_out", "W_out"])

        return helper.make_graph(
            [concat_node, resize_node],
            self.op_name,
            [input_vi, scale_vi],
            [output_vi],
            initializer=[nc_scales, roi],
        )
