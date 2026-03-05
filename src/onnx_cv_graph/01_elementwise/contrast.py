"""コントラスト調整の ONNX グラフ定義."""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class ContrastOp(OnnxGraphOp):
    """コントラスト調整.

    center を基準に contrast 倍のスケーリングを行う.
    output = (input - center) * contrast + center をクリップ.
    contrast > 1.0 でコントラスト強調、< 1.0 で低減.
    ノード構成は Sub → Mul → Add → Clip の4ノード.
    """

    @property
    def op_name(self) -> str:
        return "contrast"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("contrast", TensorProto.FLOAT, [1]),
            ("center", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {
            "contrast": (0.0, 3.0, 1.0),
            "center": (0.0, 1.0, 0.5),
        }

    def build_graph(self) -> GraphProto:
        # Clip 用の定数
        clip_min = numpy_helper.from_array(
            np.array(0.0, dtype=np.float32), name="clip_min"
        )
        clip_max = numpy_helper.from_array(
            np.array(1.0, dtype=np.float32), name="clip_max"
        )

        # Sub: input - center
        sub_node = helper.make_node(
            "Sub", inputs=["input", "center"], outputs=["centered"]
        )

        # Mul: (input - center) * contrast
        mul_node = helper.make_node(
            "Mul", inputs=["centered", "contrast"], outputs=["scaled"]
        )

        # Add: scaled + center
        add_node = helper.make_node(
            "Add", inputs=["scaled", "center"], outputs=["adjusted"]
        )

        # Clip: [0, 1] に制限
        clip_node = helper.make_node(
            "Clip", inputs=["adjusted", "clip_min", "clip_max"], outputs=["output"]
        )

        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )
        contrast_vi = helper.make_tensor_value_info(
            "contrast", TensorProto.FLOAT, [1]
        )
        center_vi = helper.make_tensor_value_info(
            "center", TensorProto.FLOAT, [1]
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )

        graph = helper.make_graph(
            [sub_node, mul_node, add_node, clip_node],
            self.op_name,
            [input_vi, contrast_vi, center_vi],
            [output_vi],
            initializer=[clip_min, clip_max],
        )
        return graph
