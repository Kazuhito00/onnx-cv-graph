"""HSV チャネル抽出の ONNX グラフ定義.

RGB→HSV 変換後、H/S/V いずれかを3ch に複製して出力する.
HSV は全て [0, 1] に正規化 (H: 0°~360° → 0~1, S: 0~1, V: 0~1).

HSV 変換アルゴリズム (OpenCV 準拠):
  V = max(R, G, B)
  S = (V - min(R, G, B)) / (V + eps)
  H = 60° × 条件分岐 / (V - min + eps), 負なら +360°, 最後に /360 で [0,1]
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class HsvExtractOp(OnnxGraphOp):
    """HSV チャネル抽出.

    RGB→HSV 変換 → 指定チャネル (h/s/v) を 3ch に複製して出力.
    """

    def __init__(self, channel: str = "h"):
        self._channel = channel

    @property
    def op_name(self) -> str:
        return f"hsv_{self._channel}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        return [cls("h"), cls("s"), cls("v")]

    def build_graph(self) -> GraphProto:
        # 定数
        eps = np.array([1e-7], dtype=np.float32)
        eps_init = numpy_helper.from_array(eps, name="eps")
        zero = np.array([0.0], dtype=np.float32)
        zero_init = numpy_helper.from_array(zero, name="zero")
        one = np.array([1.0], dtype=np.float32)
        one_init = numpy_helper.from_array(one, name="one")
        sixty = np.array([60.0], dtype=np.float32)
        sixty_init = numpy_helper.from_array(sixty, name="sixty")
        three_sixty = np.array([360.0], dtype=np.float32)
        three_sixty_init = numpy_helper.from_array(three_sixty, name="three_sixty")
        two = np.array([2.0], dtype=np.float32)
        two_init = numpy_helper.from_array(two, name="two")
        four = np.array([4.0], dtype=np.float32)
        four_init = numpy_helper.from_array(four, name="four")

        # チャネル分離用 indices
        idx_r = np.array([0], dtype=np.int64)
        idx_g = np.array([1], dtype=np.int64)
        idx_b = np.array([2], dtype=np.int64)
        idx_r_init = numpy_helper.from_array(idx_r, name="idx_r")
        idx_g_init = numpy_helper.from_array(idx_g, name="idx_g")
        idx_b_init = numpy_helper.from_array(idx_b, name="idx_b")

        # 3ch 拡張用
        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        initializers = [
            eps_init, zero_init, one_init, sixty_init, three_sixty_init,
            two_init, four_init,
            idx_r_init, idx_g_init, idx_b_init, expand_init,
        ]

        nodes = []

        # 1. チャネル分離: (N, 3, H, W) → R, G, B 各 (N, 1, H, W)
        nodes.append(helper.make_node("Gather", ["input", "idx_r"], ["R"], axis=1))
        nodes.append(helper.make_node("Gather", ["input", "idx_g"], ["G"], axis=1))
        nodes.append(helper.make_node("Gather", ["input", "idx_b"], ["B"], axis=1))

        # 2. V = max(R, G, B), Vmin = min(R, G, B)
        nodes.append(helper.make_node("Max", ["R", "G", "B"], ["V"]))
        nodes.append(helper.make_node("Min", ["R", "G", "B"], ["Vmin"]))

        # 3. diff = V - Vmin
        nodes.append(helper.make_node("Sub", ["V", "Vmin"], ["diff"]))

        # 4. S = diff / (V + eps)
        nodes.append(helper.make_node("Add", ["V", "eps"], ["V_safe"]))
        nodes.append(helper.make_node("Div", ["diff", "V_safe"], ["S"]))

        # 5. H の計算 (条件分岐)
        # diff_safe = diff + eps (ゼロ除算回避)
        nodes.append(helper.make_node("Add", ["diff", "eps"], ["diff_safe"]))

        # H 候補:
        #   V == R: h_raw = (G - B) / diff_safe
        #   V == G: h_raw = 2 + (B - R) / diff_safe
        #   V == B: h_raw = 4 + (R - G) / diff_safe

        # case R
        nodes.append(helper.make_node("Sub", ["G", "B"], ["gb"]))
        nodes.append(helper.make_node("Div", ["gb", "diff_safe"], ["h_r"]))

        # case G
        nodes.append(helper.make_node("Sub", ["B", "R"], ["br"]))
        nodes.append(helper.make_node("Div", ["br", "diff_safe"], ["br_norm"]))
        nodes.append(helper.make_node("Add", ["two", "br_norm"], ["h_g"]))

        # case B
        nodes.append(helper.make_node("Sub", ["R", "G"], ["rg"]))
        nodes.append(helper.make_node("Div", ["rg", "diff_safe"], ["rg_norm"]))
        nodes.append(helper.make_node("Add", ["four", "rg_norm"], ["h_b"]))

        # 条件選択: V == B → h_b, V == G → h_g, else h_r
        nodes.append(helper.make_node("Equal", ["V", "B"], ["is_b"]))
        nodes.append(helper.make_node("Where", ["is_b", "h_b", "h_r"], ["h_tmp1"]))
        nodes.append(helper.make_node("Equal", ["V", "G"], ["is_g"]))
        nodes.append(helper.make_node("Where", ["is_g", "h_g", "h_tmp1"], ["h_raw"]))

        # h_raw * 60 → 度数
        nodes.append(helper.make_node("Mul", ["h_raw", "sixty"], ["h_deg"]))

        # 負なら +360
        nodes.append(helper.make_node("Less", ["h_deg", "zero"], ["h_neg"]))
        nodes.append(helper.make_node("Add", ["h_deg", "three_sixty"], ["h_plus360"]))
        nodes.append(helper.make_node("Where", ["h_neg", "h_plus360", "h_deg"], ["h_deg_pos"]))

        # [0, 360] → [0, 1]
        nodes.append(helper.make_node("Div", ["h_deg_pos", "three_sixty"], ["H_raw"]))

        # diff == 0 (無彩色) なら H = 0
        nodes.append(helper.make_node("Equal", ["diff", "zero"], ["is_gray"]))
        nodes.append(helper.make_node("Where", ["is_gray", "zero", "H_raw"], ["H"]))

        # 6. 指定チャネルを選択して 3ch 拡張
        target = {"h": "H", "s": "S", "v": "V"}[self._channel]
        nodes.append(helper.make_node("Clip", [target, "zero", "one"], ["clipped"]))
        nodes.append(helper.make_node("Expand", ["clipped", "expand_shape"], ["output"]))

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes,
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=initializers,
        )
