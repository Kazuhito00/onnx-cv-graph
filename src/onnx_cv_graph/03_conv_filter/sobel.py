"""Sobel エッジ検出の ONNX グラフ定義.

グレースケール化 → Pad → Conv(Sobel X) + Conv(Sobel Y) → Abs → Add → Clip(0,1).
出力は |Gx| + |Gy| で近似したエッジ強度を 3ch に複製.

Sobel カーネル:
  Kx = [[-1,0,1],[-2,0,2],[-1,0,1]]
  Ky = [[-1,-2,-1],[0,0,0],[1,2,1]]
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class SobelOp(OnnxGraphOp):
    """Sobel エッジ検出.

    グレースケール → Pad → Conv(Kx), Conv(Ky) → Abs → Add → Clip → 3ch 拡張.
    """

    @property
    def op_name(self) -> str:
        return "sobel"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        # 輝度重み (グレースケール化用)
        luma = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        luma_init = numpy_helper.from_array(luma, name="luma")
        axes = np.array([1], dtype=np.int64)
        axes_init = numpy_helper.from_array(axes, name="axes")

        # Sobel カーネル: X,Y を統合 (2, 1, 3, 3)
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
        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        nodes = [
            # グレースケール化: (N,3,H,W) → (N,1,H,W)
            helper.make_node("Mul", ["input", "luma"], ["weighted"]),
            helper.make_node("ReduceSum", ["weighted", "axes"], ["gray"], keepdims=1),
            # Pad
            helper.make_node("Pad", ["gray", "pads"], ["padded"], mode="reflect"),
            # Sobel X, Y (統合 Conv)
            helper.make_node("Conv", ["padded", "kxy"], ["gxy"]),
            helper.make_node("Split", ["gxy"], ["gx", "gy"], axis=1),
            # |Gx| + |Gy|
            helper.make_node("Abs", ["gx"], ["abs_gx"]),
            helper.make_node("Abs", ["gy"], ["abs_gy"]),
            helper.make_node("Add", ["abs_gx", "abs_gy"], ["edge"]),
            # Clip → 3ch
            helper.make_node("Clip", ["edge", "zero", "one"], ["clipped"]),
            helper.make_node("Expand", ["clipped", "expand_shape"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
            initializer=[luma_init, axes_init, kxy_init, pads_init,
                         zero_init, one_init, expand_init],
        )
