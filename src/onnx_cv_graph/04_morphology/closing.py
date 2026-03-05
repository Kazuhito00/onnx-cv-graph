"""クロージング (closing) の ONNX グラフ定義.

膨張 → 収縮 の順で適用するモルフォロジー演算.
穴埋めに有効.
"""

from typing import List

from onnx import GraphProto, TensorProto, helper

from src.base import OnnxGraphOp, TensorSpec


class ClosingOp(OnnxGraphOp):
    """クロージング (closing) = 膨張 → 収縮.

    入力 (N,3,H,W) float32 に対し、膨張 (MaxPool) → 収縮 (Neg→MaxPool→Neg) で
    (N,3,H,W) float32 を出力する.
    """

    def __init__(self, kernel_size: int = 3):
        """カーネルサイズを指定して初期化する."""
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        return f"closing_{self._kernel_size}x{self._kernel_size}"

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

        # 膨張: MaxPool
        dilate_pool = helper.make_node(
            "MaxPool",
            inputs=["input"],
            outputs=["dilated"],
            kernel_shape=[k, k],
            pads=[pad, pad, pad, pad],
        )

        # 収縮: Neg → MaxPool → Neg
        neg1_node = helper.make_node("Neg", inputs=["dilated"], outputs=["neg_dilated"])
        erode_pool = helper.make_node(
            "MaxPool",
            inputs=["neg_dilated"],
            outputs=["neg_eroded"],
            kernel_shape=[k, k],
            pads=[pad, pad, pad, pad],
        )
        neg2_node = helper.make_node("Neg", inputs=["neg_eroded"], outputs=["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [dilate_pool, neg1_node, erode_pool, neg2_node],
            self.op_name,
            [input_vi],
            [output_vi],
        )
