"""Difference of Gaussians (DoG) エッジ検出の ONNX グラフ定義.

2つの異なるサイズのガウシアンぼかしの差分でエッジを検出する.
グレースケール化 → Pad → Conv(σ小) / Conv(σ大) → Sub → Abs → 正規化 → Clip → 3ch.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec

from .gaussian_blur import _gaussian_kernel_2d


class DogOp(OnnxGraphOp):
    """Difference of Gaussians (DoG) エッジ検出.

    2つの異なるカーネルサイズのガウシアンぼかしの差分を取り、
    エッジを検出する。出力は Abs → 正規化 → Clip(0,1) → 3ch 拡張.
    """

    def __init__(self, k1: int = 3, k2: int = 5):
        self._k1 = k1
        self._k2 = k2

    @property
    def op_name(self) -> str:
        return f"dog_{self._k1}x{self._k1}_{self._k2}x{self._k2}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        return [cls(3, 5), cls(3, 7), cls(5, 7)]

    def build_graph(self) -> GraphProto:
        k1, k2 = self._k1, self._k2
        pad1 = k1 // 2
        pad2 = k2 // 2

        # 輝度重み (グレースケール化用)
        luma = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        luma_init = numpy_helper.from_array(luma, name="luma")
        axes = np.array([1], dtype=np.int64)
        axes_init = numpy_helper.from_array(axes, name="axes")

        # ガウシアンカーネル (1ch 用: 1, 1, k, k)
        g1 = _gaussian_kernel_2d(k1).reshape(1, 1, k1, k1)
        g2 = _gaussian_kernel_2d(k2).reshape(1, 1, k2, k2)
        g1_init = numpy_helper.from_array(g1, name="kernel1")
        g2_init = numpy_helper.from_array(g2, name="kernel2")

        # パディング
        pads1 = np.array([0, 0, pad1, pad1, 0, 0, pad1, pad1], dtype=np.int64)
        pads1_init = numpy_helper.from_array(pads1, name="pads1")
        pads2 = np.array([0, 0, pad2, pad2, 0, 0, pad2, pad2], dtype=np.int64)
        pads2_init = numpy_helper.from_array(pads2, name="pads2")

        # 正規化・クリップ用定数
        # DoG の最大理論値はカーネル依存だが、Sobel と同様に適切にスケーリング
        # ガウシアン差分の最大値は約 0.25 程度なので 4.0 倍でスケーリング
        div_val = np.array([0.25], dtype=np.float32)
        div_init = numpy_helper.from_array(div_val, name="div_val")
        zero = np.array([0.0], dtype=np.float32)
        zero_init = numpy_helper.from_array(zero, name="zero")
        one = np.array([1.0], dtype=np.float32)
        one_init = numpy_helper.from_array(one, name="one")
        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        nodes = [
            # グレースケール化: (N,3,H,W) → (N,1,H,W)
            helper.make_node("Mul", ["input", "luma"], ["weighted"]),
            helper.make_node("ReduceSum", ["weighted", "axes"], ["gray"], keepdims=1),
            # σ小 のガウシアンぼかし
            helper.make_node("Pad", ["gray", "pads1"], ["padded1"], mode="reflect"),
            helper.make_node("Conv", ["padded1", "kernel1"], ["blur1"]),
            # σ大 のガウシアンぼかし
            helper.make_node("Pad", ["gray", "pads2"], ["padded2"], mode="reflect"),
            helper.make_node("Conv", ["padded2", "kernel2"], ["blur2"]),
            # 差分 → Abs → 正規化
            helper.make_node("Sub", ["blur1", "blur2"], ["diff"]),
            helper.make_node("Abs", ["diff"], ["abs_diff"]),
            helper.make_node("Div", ["abs_diff", "div_val"], ["scaled"]),
            # Clip → 3ch 拡張
            helper.make_node("Clip", ["scaled", "zero", "one"], ["clipped"]),
            helper.make_node("Expand", ["clipped", "expand_shape"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
            initializer=[
                luma_init, axes_init, g1_init, g2_init,
                pads1_init, pads2_init, div_init,
                zero_init, one_init, expand_init,
            ],
        )
