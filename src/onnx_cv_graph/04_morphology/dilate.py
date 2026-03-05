"""膨張 (dilation) の ONNX グラフ定義.

MaxPool で矩形カーネルの膨張を実現する.
パディングは入力の端ピクセルを繰り返す (edge パディング相当).
"""

from typing import List

from onnx import GraphProto, TensorProto, helper

from src.base import OnnxGraphOp, TensorSpec


class DilateOp(OnnxGraphOp):
    """膨張 (dilation).

    入力 (N,3,H,W) float32 に対し、指定カーネルサイズの MaxPool を適用して
    (N,3,H,W) float32 を出力する.
    """

    def __init__(self, kernel_size: int = 3):
        """カーネルサイズを指定して初期化する."""
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        return f"dilate_{self._kernel_size}x{self._kernel_size}"

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

        # MaxPool: kernel_shape と pads でパディング込みの膨張を実現
        maxpool_node = helper.make_node(
            "MaxPool",
            inputs=["input"],
            outputs=["output"],
            kernel_shape=[k, k],
            pads=[pad, pad, pad, pad],
        )

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [maxpool_node],
            self.op_name,
            [input_vi],
            [output_vi],
        )
