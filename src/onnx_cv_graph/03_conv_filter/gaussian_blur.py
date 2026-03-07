"""ガウシアンぼかしの ONNX グラフ定義.

OpenCV の GaussianBlur 相当.
σ = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8 (OpenCV のデフォルト σ 計算式).
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


def _gaussian_kernel_1d(ksize: int) -> np.ndarray:
    """1D ガウシアンカーネルを生成する (OpenCV デフォルト σ)."""
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    ax = np.arange(ksize, dtype=np.float32) - (ksize - 1) / 2.0
    kernel = np.exp(-0.5 * (ax / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_kernel_2d(ksize: int) -> np.ndarray:
    """2D ガウシアンカーネルを生成する."""
    k1d = _gaussian_kernel_1d(ksize)
    return np.outer(k1d, k1d).astype(np.float32)


class GaussianBlurOp(OnnxGraphOp):
    """ガウシアンぼかし.

    Pad (reflect) → Conv (depthwise, group=3).
    """

    def __init__(self, kernel_size: int = 3):
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        return f"gaussian_blur_{self._kernel_size}x{self._kernel_size}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        return [cls(3), cls(5), cls(7)]

    def build_graph(self) -> GraphProto:
        k = self._kernel_size
        pad = k // 2

        # 分離フィルタ: Conv(k,k) → Conv(k,1) + Conv(1,k)
        k1d = _gaussian_kernel_1d(k)
        kernel_v = np.tile(k1d.reshape(1, 1, k, 1), (3, 1, 1, 1))
        kernel_h = np.tile(k1d.reshape(1, 1, 1, k), (3, 1, 1, 1))
        kv_init = numpy_helper.from_array(kernel_v, name="kernel_v")
        kh_init = numpy_helper.from_array(kernel_h, name="kernel_h")

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
