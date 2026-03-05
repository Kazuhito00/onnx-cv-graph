"""Laplacian エッジ検出の ONNX グラフ定義.

単一カーネルで2次微分 (エッジ) を検出する.
Laplacian カーネル (8近傍):
  [[-1,-1,-1],
   [-1, 8,-1],
   [-1,-1,-1]]
出力は Abs → 正規化 → Clip(0,1).
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class LaplacianOp(OnnxGraphOp):
    """Laplacian エッジ検出.

    グレースケール → Pad → Conv(Laplacian) → Abs → Div(8) → Clip → 3ch 拡張.
    最大応答は中心値 8 (全周囲が逆) なので /8 で正規化.
    """

    @property
    def op_name(self) -> str:
        return "laplacian"

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

        k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32).reshape(1, 1, 3, 3)
        k_init = numpy_helper.from_array(k, name="kernel")

        pads = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
        pads_init = numpy_helper.from_array(pads, name="pads")
        zero = np.array([0.0], dtype=np.float32)
        zero_init = numpy_helper.from_array(zero, name="zero")
        one = np.array([1.0], dtype=np.float32)
        one_init = numpy_helper.from_array(one, name="one")
        norm = np.array([8.0], dtype=np.float32)
        norm_init = numpy_helper.from_array(norm, name="norm")
        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        nodes = [
            helper.make_node("Mul", ["input", "luma"], ["weighted"]),
            helper.make_node("ReduceSum", ["weighted", "axes"], ["gray"], keepdims=1),
            helper.make_node("Pad", ["gray", "pads"], ["padded"], mode="reflect"),
            helper.make_node("Conv", ["padded", "kernel"], ["conv_out"]),
            helper.make_node("Abs", ["conv_out"], ["abs_out"]),
            helper.make_node("Div", ["abs_out", "norm"], ["normed"]),
            helper.make_node("Clip", ["normed", "zero", "one"], ["clipped"]),
            helper.make_node("Expand", ["clipped", "expand_shape"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
            initializer=[luma_init, axes_init, k_init, pads_init,
                         zero_init, one_init, norm_init, expand_init],
        )
