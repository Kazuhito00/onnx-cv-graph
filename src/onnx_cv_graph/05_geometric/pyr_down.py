"""ピラミッドダウンの ONNX グラフ定義.

OpenCV の pyrDown 相当.
ガウシアンぼかし → 半分にリサイズ.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


def _gaussian_kernel_2d_5x5() -> np.ndarray:
    """pyrDown 用 5×5 ガウシアンカーネル (OpenCV デフォルト)."""
    # OpenCV pyrDown は固定 5×5 カーネル:
    # [1, 4, 6, 4, 1] / 16 の外積 / 16 = / 256
    k1d = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    k2d = np.outer(k1d, k1d)
    return (k2d / k2d.sum()).astype(np.float32)


class PyrDownOp(OnnxGraphOp):
    """ピラミッドダウン = ガウシアンぼかし (5×5) → ½ リサイズ.

    Pad → Conv (5×5 Gaussian, group=3) → Resize (scale=0.5).
    """

    @property
    def op_name(self) -> str:
        return "pyr_down"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H_out", "W_out"])]

    def build_graph(self) -> GraphProto:
        k = 5
        pad = k // 2

        g2d = _gaussian_kernel_2d_5x5()
        kernel = np.stack([g2d] * 3).reshape(3, 1, k, k)
        kernel_init = numpy_helper.from_array(kernel, name="kernel")

        pads = np.array([0, 0, pad, pad, 0, 0, pad, pad], dtype=np.int64)
        pads_init = numpy_helper.from_array(pads, name="pads")

        scales = np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float32)
        scales_init = numpy_helper.from_array(scales, name="scales")
        roi = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi")

        nodes = [
            helper.make_node("Pad", ["input", "pads"], ["padded"], mode="reflect"),
            helper.make_node("Conv", ["padded", "kernel"], ["blurred"], group=3),
            helper.make_node("Resize", ["blurred", "roi", "scales"], ["output"], mode="linear"),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H_out", "W_out"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
            initializer=[kernel_init, pads_init, scales_init, roi],
        )
