"""パディングの ONNX グラフ定義.

OpenCV の copyMakeBorder 相当.
画像の上下左右に指定ピクセル数のパディングを追加する.
パディングモード: reflect (端ピクセルを鏡像反転).
パラメータは比率 [0, 1] で指定し、H/W に乗算して実ピクセル数を算出する.
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class PaddingReflectOp(OnnxGraphOp):
    """パディング (reflect モード).

    Shape → Mul(ratio) → Floor → Cast → Concat → Pad(reflect).
    """

    @property
    def op_name(self) -> str:
        return "padding_reflect"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("pad_ratio", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N_out", 3, "H_out", "W_out"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"pad_ratio": (0.0, 0.5, 0.1)}

    def build_graph(self) -> GraphProto:
        # 定数
        zero_pads = numpy_helper.from_array(
            np.array([0, 0], dtype=np.int64), name="zero_pads",
        )
        h_idx = numpy_helper.from_array(np.array([2], dtype=np.int64), name="h_idx")
        w_idx = numpy_helper.from_array(np.array([3], dtype=np.int64), name="w_idx")

        nodes = [
            # 入力の H, W を取得
            helper.make_node("Shape", ["input"], ["shape"]),
            helper.make_node("Gather", ["shape", "h_idx"], ["H_i64"]),
            helper.make_node("Gather", ["shape", "w_idx"], ["W_i64"]),

            # int64 → float32
            helper.make_node("Cast", ["H_i64"], ["H_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["W_i64"], ["W_f"], to=TensorProto.FLOAT),

            # pad_h = Floor(H * pad_ratio), pad_w = Floor(W * pad_ratio)
            helper.make_node("Mul", ["H_f", "pad_ratio"], ["pad_h_f"]),
            helper.make_node("Floor", ["pad_h_f"], ["pad_h_floor"]),
            helper.make_node("Cast", ["pad_h_floor"], ["pad_h"], to=TensorProto.INT64),

            helper.make_node("Mul", ["W_f", "pad_ratio"], ["pad_w_f"]),
            helper.make_node("Floor", ["pad_w_f"], ["pad_w_floor"]),
            helper.make_node("Cast", ["pad_w_floor"], ["pad_w"], to=TensorProto.INT64),

            # pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]
            # Concat: [zero_pads(0,0), pad_h, pad_w, zero_pads(0,0), pad_h, pad_w]
            helper.make_node(
                "Concat",
                ["zero_pads", "pad_h", "pad_w", "zero_pads", "pad_h", "pad_w"],
                ["pads"],
                axis=0,
            ),

            # Pad (reflect)
            helper.make_node("Pad", ["input", "pads"], ["output"], mode="reflect"),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        ratio_vi = helper.make_tensor_value_info("pad_ratio", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N_out", 3, "H_out", "W_out"])

        return helper.make_graph(
            nodes, self.op_name,
            [input_vi, ratio_vi], [output_vi],
            initializer=[zero_pads, h_idx, w_idx],
        )
