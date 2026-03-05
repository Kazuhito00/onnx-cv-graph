"""Scharr エッジ検出の ONNX グラフ定義.

Sobel と同一構造でカーネルのみ異なる.
Scharr カーネル (Sobel より高精度な勾配近似):
  Kx = [[-3,0,3],[-10,0,10],[-3,0,3]]
  Ky = [[-3,-10,-3],[0,0,0],[3,10,3]]
出力は |Gx|+|Gy| を正規化して [0,1] にクリップ.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class ScharrOp(OnnxGraphOp):
    """Scharr エッジ検出.

    グレースケール → Pad → Conv(Kx), Conv(Ky) → Abs → Add → Div(32) → Clip → 3ch 拡張.
    Scharr の最大応答は 3+10+3=16 (片方向) なので合計最大 32. /32 で正規化.
    """

    @property
    def op_name(self) -> str:
        return "scharr"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        luma = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        luma_init = numpy_helper.from_array(luma, name="luma")
        axes = np.array([1], dtype=np.int64)
        axes_init = numpy_helper.from_array(axes, name="axes")

        kx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32).reshape(1, 1, 3, 3)
        ky = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32).reshape(1, 1, 3, 3)
        kx_init = numpy_helper.from_array(kx, name="kx")
        ky_init = numpy_helper.from_array(ky, name="ky")

        pads = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
        pads_init = numpy_helper.from_array(pads, name="pads")
        zero = np.array([0.0], dtype=np.float32)
        zero_init = numpy_helper.from_array(zero, name="zero")
        one = np.array([1.0], dtype=np.float32)
        one_init = numpy_helper.from_array(one, name="one")
        norm = np.array([32.0], dtype=np.float32)
        norm_init = numpy_helper.from_array(norm, name="norm")
        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        nodes = [
            helper.make_node("Mul", ["input", "luma"], ["weighted"]),
            helper.make_node("ReduceSum", ["weighted", "axes"], ["gray"], keepdims=1),
            helper.make_node("Pad", ["gray", "pads"], ["padded"], mode="reflect"),
            helper.make_node("Conv", ["padded", "kx"], ["gx"]),
            helper.make_node("Conv", ["padded", "ky"], ["gy"]),
            helper.make_node("Abs", ["gx"], ["abs_gx"]),
            helper.make_node("Abs", ["gy"], ["abs_gy"]),
            helper.make_node("Add", ["abs_gx", "abs_gy"], ["edge"]),
            helper.make_node("Div", ["edge", "norm"], ["normed"]),
            helper.make_node("Clip", ["normed", "zero", "one"], ["clipped"]),
            helper.make_node("Expand", ["clipped", "expand_shape"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
            initializer=[luma_init, axes_init, kx_init, ky_init, pads_init,
                         zero_init, one_init, norm_init, expand_init],
        )
