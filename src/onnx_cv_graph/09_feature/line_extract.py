"""直線抽出 (形態学的) の ONNX グラフ定義.

形態学的 opening (収縮→膨張) を横長/縦長カーネルで行い、直線成分を抽出する.
水平 (horizontal) / 垂直 (vertical) の2バリアント.
line_length はカーネル長で、グラフ定義時に固定する.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class LineExtractOp(OnnxGraphOp):
    """直線抽出 (形態学的 opening).

    グレースケール化 → Erode (NegMaxPool) → Dilate (NegMaxPool) で
    指定方向の直線成分を抽出する.
    カーネルは水平 (1×L) または垂直 (L×1).
    """

    def __init__(self, direction: str = "horizontal", line_length: int = 15):
        self._direction = direction
        self._line_length = line_length

    @property
    def op_name(self) -> str:
        d = "h" if self._direction == "horizontal" else "v"
        return f"line_extract_{d}_{self._line_length}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        """水平/垂直 × line_length=15 の2バリアント."""
        return [cls("horizontal", 15), cls("vertical", 15)]

    def build_graph(self) -> GraphProto:
        L = self._line_length
        pad_h = 0
        pad_w = 0

        if self._direction == "horizontal":
            kernel_shape = [1, L]
            pad_w = L // 2
        else:
            kernel_shape = [L, 1]
            pad_h = L // 2

        # 定数
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        weights_init = numpy_helper.from_array(weights, name="luma_weights")
        axes = np.array([1], dtype=np.int64)
        axes_init = numpy_helper.from_array(axes, name="axes")

        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_shape_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        neg_one = np.array([-1.0], dtype=np.float32)
        neg_one_init = numpy_helper.from_array(neg_one, name="neg_one")

        zero = np.array([0.0], dtype=np.float32)
        one = np.array([1.0], dtype=np.float32)
        zero_init = numpy_helper.from_array(zero, name="zero")
        one_init = numpy_helper.from_array(one, name="one")

        initializers = [weights_init, axes_init, expand_shape_init, neg_one_init, zero_init, one_init]

        nodes = []

        # 1. グレースケール化
        nodes.append(helper.make_node("Mul", ["input", "luma_weights"], ["weighted"]))
        nodes.append(helper.make_node("ReduceSum", ["weighted", "axes"], ["gray"], keepdims=1))

        # 2. Erode (収縮): neg → MaxPool → neg
        #    MinPool は ONNX にないので Neg → MaxPool → Neg で代替
        nodes.append(helper.make_node("Mul", ["gray", "neg_one"], ["neg_gray"]))
        nodes.append(helper.make_node(
            "MaxPool", ["neg_gray"], ["neg_eroded"],
            kernel_shape=kernel_shape,
            pads=[pad_h, pad_w, pad_h, pad_w],
        ))
        nodes.append(helper.make_node("Mul", ["neg_eroded", "neg_one"], ["eroded"]))

        # 3. Dilate (膨張): MaxPool
        nodes.append(helper.make_node(
            "MaxPool", ["eroded"], ["opened"],
            kernel_shape=kernel_shape,
            pads=[pad_h, pad_w, pad_h, pad_w],
        ))

        # 4. Clip(0, 1) → 3ch 拡張
        nodes.append(helper.make_node("Clip", ["opened", "zero", "one"], ["clipped"]))
        nodes.append(helper.make_node("Expand", ["clipped", "expand_shape"], ["output"]))

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes,
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=initializers,
        )
