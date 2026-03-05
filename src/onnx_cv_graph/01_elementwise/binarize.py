"""閾値2値化の ONNX グラフ定義."""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class BinarizeOp(OnnxGraphOp):
    """閾値2値化.

    入力 (N,3,H,W) float32 RGB 画像を内部でグレースケール化し、
    推論時に渡される閾値テンソルで2値化して (N,3,H,W) float32 を出力する.
    gray > threshold → 1.0, それ以外 → 0.0. 3チャネル同一値でチェーン合成に対応.

    ノード構成: Mul → ReduceSum → Greater → Cast → Expand の5ノード.
    """

    @property
    def op_name(self) -> str:
        return "binarize"

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
        # ITU-R BT.601 輝度重み (R, G, B)
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        weights_init = numpy_helper.from_array(weights, name="luma_weights")

        # ReduceSum の axes 入力 (opset 13+ ではテンソルとして渡す)
        axes = np.array([1], dtype=np.int64)
        axes_init = numpy_helper.from_array(axes, name="axes")

        # Expand 用の shape: チャネル次元を 3 に拡張するための [1, 3, 1, 1]
        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_shape_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        # Mul: 入力にチャネルごとの重みをブロードキャスト乗算
        mul_node = helper.make_node("Mul", inputs=["input", "luma_weights"], outputs=["weighted"])

        # ReduceSum: チャネル軸 (axis=1) で合計 → (N,1,H,W)
        reduce_node = helper.make_node(
            "ReduceSum",
            inputs=["weighted", "axes"],
            outputs=["gray"],
            keepdims=1,
        )

        # Greater: gray > threshold → bool マスク
        greater_node = helper.make_node("Greater", inputs=["gray", "threshold"], outputs=["mask"])

        # Cast: bool → float32
        cast_node = helper.make_node("Cast", inputs=["mask"], outputs=["bin_1ch"], to=TensorProto.FLOAT)

        # Expand: (N,1,H,W) → (N,3,H,W) に拡張 (3ch 同一値)
        expand_node = helper.make_node("Expand", inputs=["bin_1ch", "expand_shape"], outputs=["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        threshold_vi = helper.make_tensor_value_info("threshold", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        graph = helper.make_graph(
            [mul_node, reduce_node, greater_node, cast_node, expand_node],
            "binarize",
            [input_vi, threshold_vi],
            [output_vi],
            initializer=[weights_init, axes_init, expand_shape_init],
        )
        return graph
