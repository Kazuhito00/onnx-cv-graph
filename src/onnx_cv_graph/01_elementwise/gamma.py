"""ガンマ補正の ONNX グラフ定義."""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class GammaOp(OnnxGraphOp):
    """ガンマ補正.

    output = input ^ gamma をクリップ.
    入力は [0, 1] 範囲を前提とするため、Pow の結果も概ね [0, 1] だが
    浮動小数点誤差を考慮して Clip(0, 1) を入れる.
    gamma < 1.0 で明るく (シャドウ持ち上げ)、> 1.0 で暗く (ハイライト抑制).
    ノード構成は Pow → Clip の2ノード.
    """

    @property
    def op_name(self) -> str:
        return "gamma"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("gamma", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"gamma": (0.1, 5.0, 1.0)}

    def build_graph(self) -> GraphProto:
        # Clip 用の定数
        clip_min = numpy_helper.from_array(
            np.array(0.0, dtype=np.float32), name="clip_min"
        )
        clip_max = numpy_helper.from_array(
            np.array(1.0, dtype=np.float32), name="clip_max"
        )

        # Pow: input ^ gamma
        pow_node = helper.make_node(
            "Pow", inputs=["input", "gamma"], outputs=["powered"]
        )

        # Clip: [0, 1] に制限
        clip_node = helper.make_node(
            "Clip", inputs=["powered", "clip_min", "clip_max"], outputs=["output"]
        )

        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )
        gamma_vi = helper.make_tensor_value_info(
            "gamma", TensorProto.FLOAT, [1]
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )

        graph = helper.make_graph(
            [pow_node, clip_node],
            self.op_name,
            [input_vi, gamma_vi],
            [output_vi],
            initializer=[clip_min, clip_max],
        )
        return graph
