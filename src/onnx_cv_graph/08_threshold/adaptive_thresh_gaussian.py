"""適応的閾値処理 (ガウス) の ONNX グラフ定義.

局所ガウス加重平均との差で2値化する. OpenCV の adaptiveThreshold(ADAPTIVE_THRESH_GAUSSIAN_C) 相当.
block_size ごとに別モデルを生成する (3/5/7).
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


def _gaussian_kernel_1d(size: int) -> np.ndarray:
    """正規化済みの1Dガウシアンカーネルを生成する."""
    sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    kernel = np.exp(-0.5 * (ax / sigma) ** 2)
    return kernel / kernel.sum()


class AdaptiveThreshGaussianOp(OnnxGraphOp):
    """適応的閾値処理 (ガウス).

    グレースケール化 → ガウシアンぼかし (Conv) → gray > (local_gauss - C) で2値化.
    """

    def __init__(self, kernel_size: int = 3):
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        return f"adaptive_thresh_gaussian_{self._kernel_size}x{self._kernel_size}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("C", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"C": (-0.5, 0.5, 0.02)}

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        return [cls(3), cls(5), cls(7)]

    def build_graph(self) -> GraphProto:
        k = self._kernel_size
        pad = k // 2

        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        weights_init = numpy_helper.from_array(weights, name="luma_weights")
        axes = np.array([1], dtype=np.int64)
        axes_init = numpy_helper.from_array(axes, name="axes")
        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_shape_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        # 分離ガウシアンカーネル: (1, 1, k, 1) + (1, 1, 1, k)
        k1d = _gaussian_kernel_1d(k)
        gauss_v_init = numpy_helper.from_array(k1d.reshape(1, 1, k, 1), name="gauss_kernel_v")
        gauss_h_init = numpy_helper.from_array(k1d.reshape(1, 1, 1, k), name="gauss_kernel_h")

        # reflect パディング
        pads = np.array([0, 0, pad, pad, 0, 0, pad, pad], dtype=np.int64)
        pads_init = numpy_helper.from_array(pads, name="pads")

        # グレースケール化
        mul_node = helper.make_node("Mul", ["input", "luma_weights"], ["weighted"])
        reduce_node = helper.make_node("ReduceSum", ["weighted", "axes"], ["gray"], keepdims=1)

        # ガウシアンぼかし: Pad → Conv(v) → Conv(h) (分離フィルタ)
        pad_node = helper.make_node("Pad", ["gray", "pads"], ["padded"], mode="reflect")
        conv_v_node = helper.make_node("Conv", ["padded", "gauss_kernel_v"], ["v_blurred"])
        conv_h_node = helper.make_node("Conv", ["v_blurred", "gauss_kernel_h"], ["local_gauss"])

        # local_gauss - C → 適応的閾値
        sub_node = helper.make_node("Sub", ["local_gauss", "C"], ["adaptive_thr"])

        # gray > adaptive_thr → bool → float
        greater_node = helper.make_node("Greater", ["gray", "adaptive_thr"], ["mask"])
        cast_node = helper.make_node("Cast", ["mask"], ["bin_1ch"], to=TensorProto.FLOAT)

        # 3ch 拡張
        expand_node = helper.make_node("Expand", ["bin_1ch", "expand_shape"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        c_vi = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [mul_node, reduce_node, pad_node, conv_v_node, conv_h_node, sub_node,
             greater_node, cast_node, expand_node],
            self.op_name,
            [input_vi, c_vi],
            [output_vi],
            initializer=[weights_init, axes_init, expand_shape_init,
                         gauss_v_init, gauss_h_init, pads_init],
        )
