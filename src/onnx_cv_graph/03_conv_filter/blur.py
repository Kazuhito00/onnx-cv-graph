"""平均ぼかし (box filter) の ONNX グラフ定義."""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class BlurOp(OnnxGraphOp):
    """平均ぼかし (box filter).

    入力 (N,3,H,W) float32 に対し、指定カーネルサイズの平均フィルタを適用して
    (N,3,H,W) float32 を出力する.
    ノード構成は Pad → Conv の2ノード.
    チャネルごとに独立して畳み込む (group=3 の depthwise convolution).
    パディングは reflect モード (OpenCV の BORDER_REFLECT_101 相当) で境界アーティファクトを軽減する.
    """

    def __init__(self, kernel_size: int = 3):
        """カーネルサイズを指定して初期化する."""
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        return f"blur_{self._kernel_size}x{self._kernel_size}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        """3×3 / 5×5 / 7×7 の3バリアントを返す."""
        return [cls(3), cls(5), cls(7)]

    def build_graph(self) -> GraphProto:
        k = self._kernel_size
        pad = k // 2

        # 平均カーネル: (3, 1, k, k) — depthwise conv 用 (group=3)
        kernel = np.ones((3, 1, k, k), dtype=np.float32) / (k * k)
        kernel_init = numpy_helper.from_array(kernel, name="kernel")

        # Pad: reflect パディングで境界を処理
        # pads 形式: [N_begin, C_begin, H_begin, W_begin, N_end, C_end, H_end, W_end]
        pads = np.array([0, 0, pad, pad, 0, 0, pad, pad], dtype=np.int64)
        pads_init = numpy_helper.from_array(pads, name="pads")

        pad_node = helper.make_node(
            "Pad",
            inputs=["input", "pads"],
            outputs=["padded"],
            mode="reflect",
        )

        # Conv: group=3 で RGB チャネル独立に畳み込み
        conv_node = helper.make_node(
            "Conv",
            inputs=["padded", "kernel"],
            outputs=["output"],
            group=3,
        )

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        graph = helper.make_graph(
            [pad_node, conv_node],
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=[kernel_init, pads_init],
        )
        return graph
