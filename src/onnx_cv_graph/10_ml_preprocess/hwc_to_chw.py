"""HWC→CHW 変換の ONNX グラフ定義.

NumPy/PIL の HWC 画像を NCHW テンソルに変換する.
入力: (N, H, W, C) → 出力: (N, C, H, W)
"""

from typing import List

from onnx import GraphProto, TensorProto, helper

from src.base import OnnxGraphOp, TensorSpec


class HwcToChwOp(OnnxGraphOp):
    """HWC→CHW 変換 = Transpose(0, 3, 1, 2)."""

    @property
    def op_name(self) -> str:
        return "hwc_to_chw"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", "H", "W", 3])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]


    def build_graph(self) -> GraphProto:
        node = helper.make_node(
            "Transpose", ["input"], ["output"], perm=[0, 3, 1, 2],
        )

        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", "H", "W", 3],
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H", "W"],
        )

        return helper.make_graph(
            [node], self.op_name, [input_vi], [output_vi],
        )
