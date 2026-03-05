"""トップハット変換の ONNX グラフ定義.

元画像 - オープニング で明るい小構造を抽出する.
output = input - opening(input)
"""

from typing import List

from onnx import GraphProto, TensorProto, helper

from src.base import OnnxGraphOp, TensorSpec


class TopHatOp(OnnxGraphOp):
    """トップハット変換 = input - opening(input).

    input - (収縮→膨張).
    """

    def __init__(self, kernel_size: int = 3):
        """カーネルサイズを指定して初期化する."""
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        return f"tophat_{self._kernel_size}x{self._kernel_size}"

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
            # オープニング: 収縮 → 膨張
            helper.make_node("Neg", ["input"], ["neg_input"]),
            helper.make_node(
                "MaxPool", ["neg_input"], ["neg_eroded"],
                kernel_shape=[k, k], pads=[pad, pad, pad, pad],
            ),
            helper.make_node("Neg", ["neg_eroded"], ["eroded"]),
            helper.make_node(
                "MaxPool", ["eroded"], ["opened"],
                kernel_shape=[k, k], pads=[pad, pad, pad, pad],
            ),
            # トップハット: input - opening
            helper.make_node("Sub", ["input", "opened"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
        )
