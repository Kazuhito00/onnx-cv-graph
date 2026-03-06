"""Kuwahara フィルタの ONNX グラフ定義."""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class KuwaharaOp(OnnxGraphOp):
    """Kuwahara フィルタ.

    各ピクセルの周辺を4つの重複サブ領域（左上・右上・左下・右下）に分割し、
    輝度の分散が最小のサブ領域の RGB 平均値を出力とする非線形平滑化フィルタ。
    エッジを保持しながらノイズを除去するため、絵画的な平滑効果が得られる。

    分散計算は ITU-R BT.601 グレースケール輝度で行い、
    RGB 各チャネルには同じ領域選択を適用する。
    カーネルサイズはグラフ定義時に固定。

    ノード構成:
        Mul+ReduceSum (グレースケール) → Mul (自乗) →
        AveragePool×4×3 (各象限の gray / gray² / RGB 平均) →
        Sub (分散) → Concat → ReduceMin → Equal×4 → Where×3 → Clip
    """

    def __init__(self, kernel_size: int = 5):
        assert kernel_size % 2 == 1 and kernel_size >= 5, \
            "kernel_size は奇数かつ 5 以上"
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        k = self._kernel_size
        return f"kuwahara_{k}x{k}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        return [cls(5), cls(7), cls(9)]

    def build_graph(self) -> GraphProto:
        k = self._kernel_size
        r = (k - 1) // 2
        sub = r + 1  # サブ領域サイズ

        nodes = []

        # --- 初期化テンソル ---
        luma_w = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        luma_w_init = numpy_helper.from_array(luma_w, "luma_w")

        axes1_init = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes1")

        clip_min_init = numpy_helper.from_array(np.float32(0.0), "clip_min")
        clip_max_init = numpy_helper.from_array(np.float32(1.0), "clip_max")

        initializers = [luma_w_init, axes1_init, clip_min_init, clip_max_init]

        # --- 1. グレースケール輝度 (N,1,H,W) ---
        nodes.append(helper.make_node("Mul", ["input", "luma_w"], ["weighted"]))
        nodes.append(helper.make_node(
            "ReduceSum", ["weighted", "axes1"], ["gray"], keepdims=1,
        ))

        # --- 2. 輝度の自乗 (N,1,H,W) ---
        nodes.append(helper.make_node("Mul", ["gray", "gray"], ["gray2"]))

        # --- 3. 各象限の AveragePool ---
        # pads = [top, left, bottom, right] (ONNX AveragePool 形式)
        quadrants = [
            ([r, r, 0, 0], "q1"),  # 左上サブ領域
            ([r, 0, 0, r], "q2"),  # 右上サブ領域
            ([0, r, r, 0], "q3"),  # 左下サブ領域
            ([0, 0, r, r], "q4"),  # 右下サブ領域
        ]

        for pads, qname in quadrants:
            pool_kwargs = dict(
                kernel_shape=[sub, sub],
                pads=pads,
                strides=[1, 1],
                count_include_pad=0,
            )

            # 輝度の象限平均 → 分散計算に使用
            nodes.append(helper.make_node(
                "AveragePool", ["gray"], [f"mean_gray_{qname}"], **pool_kwargs,
            ))
            # 輝度² の象限平均 → E[X²]
            nodes.append(helper.make_node(
                "AveragePool", ["gray2"], [f"mean_gray2_{qname}"], **pool_kwargs,
            ))
            # 分散 = E[X²] - E[X]²
            nodes.append(helper.make_node(
                "Mul",
                [f"mean_gray_{qname}", f"mean_gray_{qname}"],
                [f"mean_gray_sq_{qname}"],
            ))
            nodes.append(helper.make_node(
                "Sub",
                [f"mean_gray2_{qname}", f"mean_gray_sq_{qname}"],
                [f"var_{qname}"],
            ))

            # RGB の象限平均 (N,3,H,W) → 出力候補
            nodes.append(helper.make_node(
                "AveragePool", ["input"], [f"mean_rgb_{qname}"], **pool_kwargs,
            ))

        # --- 4. トーナメント選択で最小分散象限の RGB 平均を取得 ---
        # Equal(float) + ReduceMin は onnxruntime-web WASM で動作保証がないため
        # Less + Where のペア比較に置き換える。Less は整数・浮動小数ともに安定動作する。

        # Round 1: Q1 vs Q2
        nodes.append(helper.make_node("Less", ["var_q1", "var_q2"], ["q1_lt_q2"]))
        nodes.append(helper.make_node(
            "Where", ["q1_lt_q2", "var_q1", "var_q2"], ["min_var_12"],
        ))
        nodes.append(helper.make_node(
            "Where", ["q1_lt_q2", "mean_rgb_q1", "mean_rgb_q2"], ["mean_rgb_12"],
        ))

        # Round 1: Q3 vs Q4
        nodes.append(helper.make_node("Less", ["var_q3", "var_q4"], ["q3_lt_q4"]))
        nodes.append(helper.make_node(
            "Where", ["q3_lt_q4", "var_q3", "var_q4"], ["min_var_34"],
        ))
        nodes.append(helper.make_node(
            "Where", ["q3_lt_q4", "mean_rgb_q3", "mean_rgb_q4"], ["mean_rgb_34"],
        ))

        # Final: winner of (Q1,Q2) vs winner of (Q3,Q4)
        nodes.append(helper.make_node("Less", ["min_var_12", "min_var_34"], ["group12_wins"]))
        nodes.append(helper.make_node(
            "Where", ["group12_wins", "mean_rgb_12", "mean_rgb_34"], ["selected"],
        ))

        # --- 6. Clip [0, 1] ---
        nodes.append(helper.make_node(
            "Clip", ["selected", "clip_min", "clip_max"], ["output"],
        ))

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes,
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=initializers,
        )
