"""反転 (flip) の ONNX グラフ定義.

Slice (step=-1) で軸反転を実現する.
INT_MAX/INT_MIN をスライス境界に使い、ONNX が自動クランプする仕様を活用.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec

_INT64_MAX = np.iinfo(np.int64).max
_INT64_MIN = np.iinfo(np.int64).min


def _build_reverse_nodes(input_name: str, output_name: str, axis: int, prefix: str):
    """指定軸を Slice(step=-1) で逆順にするヘルパー.

    Returns: (nodes, initializers)
    """
    starts = numpy_helper.from_array(np.array([_INT64_MAX], dtype=np.int64), f"{prefix}_starts")
    ends = numpy_helper.from_array(np.array([_INT64_MIN], dtype=np.int64), f"{prefix}_ends")
    axes = numpy_helper.from_array(np.array([axis], dtype=np.int64), f"{prefix}_axes")
    steps = numpy_helper.from_array(np.array([-1], dtype=np.int64), f"{prefix}_steps")

    slice_node = helper.make_node(
        "Slice",
        [input_name, f"{prefix}_starts", f"{prefix}_ends", f"{prefix}_axes", f"{prefix}_steps"],
        [output_name],
    )

    nodes = [slice_node]
    inits = [starts, ends, axes, steps]
    return nodes, inits


class HFlipOp(OnnxGraphOp):
    """水平反転 (左右反転)."""

    @property
    def op_name(self) -> str:
        return "hflip"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        nodes, inits = _build_reverse_nodes("input", "output", axis=3, prefix="w")

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(nodes, self.op_name, [input_vi], [output_vi], initializer=inits)


class VFlipOp(OnnxGraphOp):
    """垂直反転 (上下反転)."""

    @property
    def op_name(self) -> str:
        return "vflip"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        nodes, inits = _build_reverse_nodes("input", "output", axis=2, prefix="h")

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(nodes, self.op_name, [input_vi], [output_vi], initializer=inits)


class HVFlipOp(OnnxGraphOp):
    """上下左右反転 (180° 反転と同等)."""

    @property
    def op_name(self) -> str:
        return "hvflip"

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
