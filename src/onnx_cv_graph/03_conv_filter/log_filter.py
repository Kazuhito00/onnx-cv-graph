"""Laplacian of Gaussian (LoG) フィルタの ONNX グラフ定義.

ガウシアンぼかしと Laplacian を1つのカーネルに統合.
LoG(x,y) = -(1/(π σ⁴)) * (1 - (x²+y²)/(2σ²)) * exp(-(x²+y²)/(2σ²))
σ は OpenCV のデフォルト計算式を使用.
5×5 / 7×7 の2バリアント.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


def _log_kernel(ksize: int) -> np.ndarray:
    """LoG カーネルを生成する."""
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    ax = np.arange(ksize, dtype=np.float64) - (ksize - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx ** 2 + yy ** 2
    s2 = sigma ** 2
    kernel = -(1.0 / (np.pi * s2 ** 2)) * (1.0 - r2 / (2.0 * s2)) * np.exp(-r2 / (2.0 * s2))
    # ゼロサムにする (DC 成分除去)
    kernel -= kernel.mean()
    return kernel.astype(np.float32)


class LogFilterOp(OnnxGraphOp):
    """Laplacian of Gaussian (LoG) フィルタ.

    グレースケール → Pad → Conv(LoG) → Abs → 正規化 → Clip → 3ch 拡張.
    """

    def __init__(self, kernel_size: int = 5):
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        return f"log_{self._kernel_size}x{self._kernel_size}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        return [cls(5), cls(7)]

    def build_graph(self) -> GraphProto:
        k = self._kernel_size
        pad = k // 2

        luma = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        luma_init = numpy_helper.from_array(luma, name="luma")
        axes = np.array([1], dtype=np.int64)
        axes_init = numpy_helper.from_array(axes, name="axes")

        log_k = _log_kernel(k).reshape(1, 1, k, k)
        # 正規化係数: カーネルの正の要素の合計を使用
        norm_val = float(np.abs(log_k).sum())
        kernel_init = numpy_helper.from_array(log_k, name="kernel")

        pads = np.array([0, 0, pad, pad, 0, 0, pad, pad], dtype=np.int64)
        pads_init = numpy_helper.from_array(pads, name="pads")
        zero = np.array([0.0], dtype=np.float32)
        zero_init = numpy_helper.from_array(zero, name="zero")
        one = np.array([1.0], dtype=np.float32)
        one_init = numpy_helper.from_array(one, name="one")
        norm = np.array([norm_val], dtype=np.float32)
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
            initializer=[luma_init, axes_init, kernel_init, pads_init,
                         zero_init, one_init, norm_init, expand_init],
        )
