"""エッジ強度 (magnitude) の ONNX グラフ定義.

Sobel X/Y の二乗和平方根: sqrt(Gx² + Gy²) → 正規化 → Clip(0,1).
|Gx|+|Gy| 近似ではなく真のユークリッド距離.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class EdgeMagnitudeOp(OnnxGraphOp):
    """エッジ強度 (magnitude).

    グレースケール → Pad → Conv(Sobel X,Y) → Mul → Add → Sqrt → Div(norm) → Clip → 3ch.
    Sobel の最大応答は 4 (片方向) なので magnitude 最大 = sqrt(16+16) ≈ 5.66.
    """

    @property
    def op_name(self) -> str:
        return "edge_magnitude"

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

        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32).reshape(1, 1, 3, 3)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32).reshape(1, 1, 3, 3)
        kxy = np.concatenate([kx, ky], axis=0)
        kxy_init = numpy_helper.from_array(kxy, name="kxy")

        pads = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
        pads_init = numpy_helper.from_array(pads, name="pads")
        zero = np.array([0.0], dtype=np.float32)
        zero_init = numpy_helper.from_array(zero, name="zero")
        one = np.array([1.0], dtype=np.float32)
        one_init = numpy_helper.from_array(one, name="one")
        # sqrt(4^2 + 4^2) ≈ 5.657
        norm = np.array([np.sqrt(32.0)], dtype=np.float32)
        norm_init = numpy_helper.from_array(norm, name="norm")
        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        nodes = [
            helper.make_node("Mul", ["input", "luma"], ["weighted"]),
            helper.make_node("ReduceSum", ["weighted", "axes"], ["gray"], keepdims=1),
            helper.make_node("Pad", ["gray", "pads"], ["padded"], mode="reflect"),
            helper.make_node("Conv", ["padded", "kxy"], ["gxy"]),
            helper.make_node("Split", ["gxy"], ["gx", "gy"], axis=1),
            # Gx² + Gy²
            helper.make_node("Mul", ["gx", "gx"], ["gx2"]),
            helper.make_node("Mul", ["gy", "gy"], ["gy2"]),
            helper.make_node("Add", ["gx2", "gy2"], ["sum_sq"]),
            helper.make_node("Sqrt", ["sum_sq"], ["mag"]),
            # 正規化 → Clip → 3ch
            helper.make_node("Div", ["mag", "norm"], ["normed"]),
            helper.make_node("Clip", ["normed", "zero", "one"], ["clipped"]),
            helper.make_node("Expand", ["clipped", "expand_shape"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
            initializer=[luma_init, axes_init, kxy_init, pads_init,
                         zero_init, one_init, norm_init, expand_init],
        )
