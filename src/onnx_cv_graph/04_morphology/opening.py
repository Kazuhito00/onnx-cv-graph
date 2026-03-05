"""オープニング (opening) の ONNX グラフ定義.

収縮 → 膨張 の順で適用するモルフォロジー演算.
ノイズ除去に有効.
"""

from typing import List

from onnx import GraphProto, TensorProto, helper

from src.base import OnnxGraphOp, TensorSpec


class OpeningOp(OnnxGraphOp):
    """オープニング (opening) = 収縮 → 膨張.

    入力 (N,3,H,W) float32 に対し、収縮 (Neg→MaxPool→Neg) → 膨張 (MaxPool) で
    (N,3,H,W) float32 を出力する.
    """

    def __init__(self, kernel_size: int = 3):
        """カーネルサイズを指定して初期化する."""
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        return f"opening_{self._kernel_size}x{self._kernel_size}"

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

        # 収縮: Neg → MaxPool → Neg
        neg1_node = helper.make_node("Neg", inputs=["input"], outputs=["neg_input"])
        erode_pool = helper.make_node(
            "MaxPool",
            inputs=["neg_input"],
            outputs=["neg_eroded"],
            kernel_shape=[k, k],
            pads=[pad, pad, pad, pad],
        )
        neg2_node = helper.make_node("Neg", inputs=["neg_eroded"], outputs=["eroded"])

        # 膨張: MaxPool
        dilate_pool = helper.make_node(
            "MaxPool",
            inputs=["eroded"],
            outputs=["output"],
            kernel_shape=[k, k],
            pads=[pad, pad, pad, pad],
        )

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [neg1_node, erode_pool, neg2_node, dilate_pool],
            self.op_name,
            [input_vi],
            [output_vi],
        )
