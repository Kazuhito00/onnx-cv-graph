"""ブラックハット変換の ONNX グラフ定義.

クロージング - 元画像 で暗い小構造を抽出する.
output = closing(input) - input
"""

from typing import List

from onnx import GraphProto, TensorProto, helper

from src.base import OnnxGraphOp, TensorSpec


class BlackHatOp(OnnxGraphOp):
    """ブラックハット変換 = closing(input) - input.

    (膨張→収縮) - input.
    """

    def __init__(self, kernel_size: int = 3):
        """カーネルサイズを指定して初期化する."""
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        return f"blackhat_{self._kernel_size}x{self._kernel_size}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        """3×3 / 5×5 の2バリアントを返す."""
        return [cls(3), cls(5)]

    def build_graph(self) -> GraphProto:
        k = self._kernel_size
        pad = k // 2

        nodes = [
            # クロージング: 膨張 → 収縮
            helper.make_node(
                "MaxPool", ["input"], ["dilated"],
                kernel_shape=[k, k], pads=[pad, pad, pad, pad],
            ),
            helper.make_node("Neg", ["dilated"], ["neg_dilated"]),
            helper.make_node(
                "MaxPool", ["neg_dilated"], ["neg_closed"],
                kernel_shape=[k, k], pads=[pad, pad, pad, pad],
            ),
            helper.make_node("Neg", ["neg_closed"], ["closed"]),
            # ブラックハット: closing - input
            helper.make_node("Sub", ["closed", "input"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
        )
