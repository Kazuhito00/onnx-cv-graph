"""セピア調変換の ONNX グラフ定義."""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class SepiaOp(OnnxGraphOp):
    """セピア調変換.

    入力 (N,3,H,W) float32 RGB に対し、セピア色変換行列を 1×1 Conv で適用する.
    出力が [0, 1] を超えうるため Clip(0, 1) で値域を制限する.
    ノード構成は Conv(1×1) → Clip の2ノード.

    セピア変換行列 (Microsoft 標準):
        R' = 0.393*R + 0.769*G + 0.189*B
        G' = 0.349*R + 0.686*G + 0.168*B
        B' = 0.272*R + 0.534*G + 0.131*B
    """

    @property
    def op_name(self) -> str:
        return "sepia"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        # セピア変換行列: 出力チャネル × 入力チャネル (3×3)
        # Conv の weight shape は (out_ch, in_ch, kH, kW) = (3, 3, 1, 1)
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],  # R'
            [0.349, 0.686, 0.168],  # G'
            [0.272, 0.534, 0.131],  # B'
        ], dtype=np.float32).reshape(3, 3, 1, 1)
        kernel_init = numpy_helper.from_array(sepia_matrix, name="sepia_kernel")

        # Clip 用の定数
        clip_min = numpy_helper.from_array(
            np.array(0.0, dtype=np.float32), name="clip_min"
        )
        clip_max = numpy_helper.from_array(
            np.array(1.0, dtype=np.float32), name="clip_max"
        )

        # Conv(1×1): 色変換行列の適用
        conv_node = helper.make_node(
            "Conv", inputs=["input", "sepia_kernel"], outputs=["conv_out"]
        )

        # Clip: [0, 1] に制限 (全白入力で R' = 1.351 等になるため)
        clip_node = helper.make_node(
            "Clip", inputs=["conv_out", "clip_min", "clip_max"], outputs=["output"]
        )

        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )

        graph = helper.make_graph(
            [conv_node, clip_node],
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=[kernel_init, clip_min, clip_max],
        )
        return graph
