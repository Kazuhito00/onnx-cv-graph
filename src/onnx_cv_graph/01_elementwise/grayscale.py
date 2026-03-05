"""グレースケール変換の ONNX グラフ定義."""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class GrayscaleOp(OnnxGraphOp):
    """RGB→グレースケール変換.

    入力 (N,3,H,W) float32 に対し、ITU-R BT.601 輝度重みで加重和を取り
    (N,3,H,W) float32 を出力する. 3チャネル同一値でチェーン合成に対応.
    ノード構成は Mul + ReduceSum + Expand の3ノード.
    """

    @property
    def op_name(self) -> str:
        return "grayscale"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

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
        # Expand: (N,1,H,W) → (N,3,H,W) に拡張 (3ch 同一値)
        expand_node = helper.make_node("Expand", inputs=["gray", "expand_shape"], outputs=["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        graph = helper.make_graph(
            [mul_node, reduce_node, expand_node],
            "grayscale",
            [input_vi],
            [output_vi],
            initializer=[weights_init, axes_init, expand_shape_init],
        )
        return graph
