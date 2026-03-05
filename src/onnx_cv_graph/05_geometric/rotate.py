"""回転 (rotation) の ONNX グラフ定義.

90°: Transpose(0,1,3,2) → 最終軸反転
180°: H軸反転 → W軸反転
270°: Transpose(0,1,3,2) → 第2空間軸反転
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec
from .flip import _build_reverse_nodes


class Rotate90Op(OnnxGraphOp):
    """90° 時計回り回転."""

    @property
    def op_name(self) -> str:
        return "rotate_90"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "W", "H"])]

    def build_graph(self) -> GraphProto:
        # Transpose: (N,C,H,W) → (N,C,W,H)
        transpose_node = helper.make_node(
            "Transpose", ["input"], ["transposed"], perm=[0, 1, 3, 2]
        )
        # 最終軸 (元の H) を反転
        rev_nodes, rev_inits = _build_reverse_nodes("transposed", "output", axis=3, prefix="r")

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "W", "H"])

        return helper.make_graph(
            [transpose_node] + rev_nodes, self.op_name, [input_vi], [output_vi],
            initializer=rev_inits,
        )


class Rotate180Op(OnnxGraphOp):
    """180° 回転 (= 上下左右反転)."""

    @property
    def op_name(self) -> str:
        return "rotate_180"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        h_nodes, h_inits = _build_reverse_nodes("input", "vflipped", axis=2, prefix="h")
        w_nodes, w_inits = _build_reverse_nodes("vflipped", "output", axis=3, prefix="w")

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            h_nodes + w_nodes, self.op_name, [input_vi], [output_vi],
            initializer=h_inits + w_inits,
        )


class Rotate270Op(OnnxGraphOp):
    """270° 時計回り回転 (= 90° 反時計回り)."""

    @property
    def op_name(self) -> str:
        return "rotate_270"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "W", "H"])]

    def build_graph(self) -> GraphProto:
        # Transpose: (N,C,H,W) → (N,C,W,H)
        transpose_node = helper.make_node(
            "Transpose", ["input"], ["transposed"], perm=[0, 1, 3, 2]
        )
        # 第2空間軸 (元の W, axis=2) を反転
        rev_nodes, rev_inits = _build_reverse_nodes("transposed", "output", axis=2, prefix="r")

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "W", "H"])

        return helper.make_graph(
            [transpose_node] + rev_nodes, self.op_name, [input_vi], [output_vi],
            initializer=rev_inits,
        )
