"""CHW→HWC 変換の ONNX グラフ定義.

NCHW テンソルを NumPy/PIL の HWC 画像に変換する.
入力: (N, C, H, W) → 出力: (N, H, W, C)
"""

from typing import List

from onnx import GraphProto, TensorProto, helper

from src.base import OnnxGraphOp, TensorSpec


class ChwToHwcOp(OnnxGraphOp):
    """CHW→HWC 変換 = Transpose(0, 2, 3, 1)."""

    @property
    def op_name(self) -> str:
        return "chw_to_hwc"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", "H", "W", 3])]


    def build_graph(self) -> GraphProto:
        node = helper.make_node(
            "Transpose", ["input"], ["output"], perm=[0, 2, 3, 1],
        )

        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", 3, "H", "W"],
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", "H", "W", 3],
        )

        return helper.make_graph(
            [node], self.op_name, [input_vi], [output_vi],
        )
