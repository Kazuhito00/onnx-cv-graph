"""明るさ調整の ONNX グラフ定義."""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class BrightnessOp(OnnxGraphOp):
    """明るさ調整.

    入力 (N,3,H,W) float32 に brightness を加算し、Clip(0,1) で値域を制限する.
    brightness > 0 で明るく、< 0 で暗くなる.
    ノード構成は Add → Clip の2ノード.
    """

    @property
    def op_name(self) -> str:
        return "brightness"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("brightness", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"brightness": (-1.0, 1.0, 0.0)}

    def build_graph(self) -> GraphProto:
        # Clip 用の定数
        clip_min = numpy_helper.from_array(
            np.array(0.0, dtype=np.float32), name="clip_min"
        )
        clip_max = numpy_helper.from_array(
            np.array(1.0, dtype=np.float32), name="clip_max"
        )

        # Add: input + brightness
        add_node = helper.make_node(
            "Add", inputs=["input", "brightness"], outputs=["added"]
        )

        # Clip: [0, 1] に制限
        clip_node = helper.make_node(
            "Clip", inputs=["added", "clip_min", "clip_max"], outputs=["output"]
        )

        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )
        brightness_vi = helper.make_tensor_value_info(
            "brightness", TensorProto.FLOAT, [1]
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )

        graph = helper.make_graph(
            [add_node, clip_node],
            self.op_name,
            [input_vi, brightness_vi],
            [output_vi],
            initializer=[clip_min, clip_max],
        )
        return graph
