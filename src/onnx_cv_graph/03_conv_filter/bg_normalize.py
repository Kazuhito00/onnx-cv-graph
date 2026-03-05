"""背景ムラ補正の ONNX グラフ定義.

ガウシアンぼかしで低周波成分 (背景ムラ) を推定し、元画像から差し引く.
  blur = GaussianBlur(input)
  output = Clip(input - blur + 0.5, 0, 1)

影除去・紙の色ムラ除去に有効. +0.5 のオフセットで中間グレーを基準にする.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec

from .gaussian_blur import _gaussian_kernel_2d


class BgNormalizeOp(OnnxGraphOp):
    """背景ムラ補正.

    Pad → Conv(Gaussian) → Sub(input - blur) → Add(0.5) → Clip(0,1).
    """

    def __init__(self, kernel_size: int = 7):
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        return f"bg_normalize_{self._kernel_size}x{self._kernel_size}"

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

        g2d = _gaussian_kernel_2d(k)
        kernel = np.stack([g2d] * 3).reshape(3, 1, k, k)
        kernel_init = numpy_helper.from_array(kernel, name="kernel")

        pads = np.array([0, 0, pad, pad, 0, 0, pad, pad], dtype=np.int64)
        pads_init = numpy_helper.from_array(pads, name="pads")

        offset = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="offset")
        zero = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="zero")
        one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")

        nodes = [
            # ガウシアンぼかし (背景推定)
            helper.make_node("Pad", ["input", "pads"], ["padded"], mode="reflect"),
            helper.make_node("Conv", ["padded", "kernel"], ["blur"], group=3),
            # input - blur + 0.5
            helper.make_node("Sub", ["input", "blur"], ["diff"]),
            helper.make_node("Add", ["diff", "offset"], ["shifted"]),
            # Clip(0, 1)
            helper.make_node("Clip", ["shifted", "zero", "one"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
            initializer=[kernel_init, pads_init, offset, zero, one],
        )
