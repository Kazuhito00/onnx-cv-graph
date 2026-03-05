"""ピラミッドアップの ONNX グラフ定義.

OpenCV の pyrUp 相当.
2倍にリサイズ → ガウシアンぼかし.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec

from .pyr_down import _gaussian_kernel_2d_5x5


class PyrUpOp(OnnxGraphOp):
    """ピラミッドアップ = ×2 リサイズ → ガウシアンぼかし (5×5).

    Resize (scale=2.0) → Pad → Conv (5×5 Gaussian, group=3).
    """

    @property
    def op_name(self) -> str:
        return "pyr_up"

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

        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
        scales_init = numpy_helper.from_array(scales, name="scales")
        roi = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi")

        zero = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="zero")
        one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")

        nodes = [
            helper.make_node("Resize", ["input", "roi", "scales"], ["upscaled"], mode="linear"),
            helper.make_node("Pad", ["upscaled", "pads"], ["padded"], mode="reflect"),
            helper.make_node("Conv", ["padded", "kernel"], ["blurred"], group=3),
            helper.make_node("Clip", ["blurred", "zero", "one"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H_out", "W_out"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
            initializer=[kernel_init, pads_init, scales_init, roi, zero, one],
        )
