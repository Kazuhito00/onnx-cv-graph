"""シャープ化フィルタの ONNX グラフ定義.

3×3 シャープカーネル:
  [[ 0, -1,  0],
   [-1,  5, -1],
   [ 0, -1,  0]]
出力が [0,1] を超えうるため Clip(0,1) で制限する.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class SharpenOp(OnnxGraphOp):
    """シャープ化.

    Pad (reflect) → Conv (depthwise) → Clip(0,1).
    """

    @property
    def op_name(self) -> str:
        return "sharpen"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        kernel = np.stack([k] * 3).reshape(3, 1, 3, 3)
        kernel_init = numpy_helper.from_array(kernel, name="kernel")

        pads = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
        pads_init = numpy_helper.from_array(pads, name="pads")
        zero = np.array([0.0], dtype=np.float32)
        zero_init = numpy_helper.from_array(zero, name="zero")
        one = np.array([1.0], dtype=np.float32)
        one_init = numpy_helper.from_array(one, name="one")

        nodes = [
            helper.make_node("Pad", ["input", "pads"], ["padded"], mode="reflect"),
            helper.make_node("Conv", ["padded", "kernel"], ["conv_out"], group=3),
            helper.make_node("Clip", ["conv_out", "zero", "one"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
            initializer=[kernel_init, pads_init, zero_init, one_init],
        )
