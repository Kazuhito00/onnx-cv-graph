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

        # 分離フィルタ: box(k,k) → box(k,1) + box(1,k)
        k1d_v = np.ones((3, 1, k, 1), dtype=np.float32) / k
        k1d_h = np.ones((3, 1, 1, k), dtype=np.float32) / k
        kv_init = numpy_helper.from_array(k1d_v, name="kernel_v")
        kh_init = numpy_helper.from_array(k1d_h, name="kernel_h")

        pads = np.array([0, 0, pad, pad, 0, 0, pad, pad], dtype=np.int64)
        pads_init = numpy_helper.from_array(pads, name="pads")

        nodes = [
            helper.make_node("Pad", ["input", "pads"], ["padded"], mode="reflect"),
            helper.make_node("Conv", ["padded", "kernel_v"], ["v_blurred"], group=3),
            helper.make_node("Conv", ["v_blurred", "kernel_h"], ["output"], group=3),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
            initializer=[kv_init, kh_init, pads_init],
        )
