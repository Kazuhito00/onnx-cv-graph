"""モルフォロジー勾配の ONNX グラフ定義.

膨張 - 収縮 でエッジ (境界) を抽出する.
output = dilate(input) - erode(input)
"""

from typing import List

from onnx import GraphProto, TensorProto, helper

from src.base import OnnxGraphOp, TensorSpec


class GradientOp(OnnxGraphOp):
    """モルフォロジー勾配 = 膨張 - 収縮.

    MaxPool(input) - (-MaxPool(-input)) → Clip(0,1).
    """

    def __init__(self, kernel_size: int = 3):
        """カーネルサイズを指定して初期化する."""
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        return f"gradient_{self._kernel_size}x{self._kernel_size}"

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
            # 膨張: MaxPool
            helper.make_node(
                "MaxPool", ["input"], ["dilated"],
                kernel_shape=[k, k], pads=[pad, pad, pad, pad],
            ),
            # 収縮: Neg → MaxPool → Neg
            helper.make_node("Neg", ["input"], ["neg_input"]),
            helper.make_node(
                "MaxPool", ["neg_input"], ["neg_eroded"],
                kernel_shape=[k, k], pads=[pad, pad, pad, pad],
            ),
            helper.make_node("Neg", ["neg_eroded"], ["eroded"]),
            # 勾配: 膨張 - 収縮
            helper.make_node("Sub", ["dilated", "eroded"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
        )
