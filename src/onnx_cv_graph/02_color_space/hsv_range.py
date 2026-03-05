"""HSV 範囲抽出の ONNX グラフ定義.

RGB→HSV 変換後、H/S/V 各チャネルが指定範囲内にあるピクセルを 1.0、
それ以外を 0.0 にするマスクを出力する. OpenCV の cvtColor + inRange 相当.

色相 (H) は循環するため h_min > h_max の場合は折り返し判定を行う:
  h_min <= h_max: h_min ≤ H ≤ h_max
  h_min > h_max:  H ≥ h_min OR H ≤ h_max  (赤領域など)

HSV は全て [0, 1] に正規化.
パラメータ: h_min, h_max, s_min, s_max, v_min, v_max (各 [0, 1])
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class HsvRangeOp(OnnxGraphOp):
    """HSV 範囲抽出.

    RGB→HSV 変換 → 各チャネルの範囲判定 → AND → 3ch マスク出力.
    色相の折り返し (h_min > h_max) に対応.
    """

    @property
    def op_name(self) -> str:
        return "hsv_range"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("h_min", TensorProto.FLOAT, [1]),
            ("h_max", TensorProto.FLOAT, [1]),
            ("s_min", TensorProto.FLOAT, [1]),
            ("s_max", TensorProto.FLOAT, [1]),
            ("v_min", TensorProto.FLOAT, [1]),
            ("v_max", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {
            "h_min": (0.0, 1.0, 0.0),
            "h_max": (0.0, 1.0, 1.0),
            "s_min": (0.0, 1.0, 0.0),
            "s_max": (0.0, 1.0, 1.0),
            "v_min": (0.0, 1.0, 0.0),
            "v_max": (0.0, 1.0, 1.0),
        }

    def build_graph(self) -> GraphProto:
        # --- 定数 ---
        eps = np.array([1e-7], dtype=np.float32)
        eps_init = numpy_helper.from_array(eps, name="eps")
        zero = np.array([0.0], dtype=np.float32)
        zero_init = numpy_helper.from_array(zero, name="zero")
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
            eps_init, zero_init, sixty_init, three_sixty_init,
            two_init, four_init,
            idx_r_init, idx_g_init, idx_b_init, expand_init,
        ]

        nodes = []

        # ===== RGB → HSV 変換 (hsv_extract.py と同一ロジック) =====

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

        # 5. H の計算
        nodes.append(helper.make_node("Add", ["diff", "eps"], ["diff_safe"]))

        # case R: h_r = (G - B) / diff_safe
        nodes.append(helper.make_node("Sub", ["G", "B"], ["gb"]))
        nodes.append(helper.make_node("Div", ["gb", "diff_safe"], ["h_r"]))

        # case G: h_g = 2 + (B - R) / diff_safe
        nodes.append(helper.make_node("Sub", ["B", "R"], ["br"]))
        nodes.append(helper.make_node("Div", ["br", "diff_safe"], ["br_norm"]))
        nodes.append(helper.make_node("Add", ["two", "br_norm"], ["h_g"]))

        # case B: h_b = 4 + (R - G) / diff_safe
        nodes.append(helper.make_node("Sub", ["R", "G"], ["rg"]))
        nodes.append(helper.make_node("Div", ["rg", "diff_safe"], ["rg_norm"]))
        nodes.append(helper.make_node("Add", ["four", "rg_norm"], ["h_b"]))

        # 条件選択
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

        # ===== 範囲判定 =====

        # S の範囲判定: s_min ≤ S ≤ s_max
        nodes.append(helper.make_node("GreaterOrEqual", ["S", "s_min"], ["s_ge"]))
        nodes.append(helper.make_node("LessOrEqual", ["S", "s_max"], ["s_le"]))
        nodes.append(helper.make_node("And", ["s_ge", "s_le"], ["s_in"]))

        # V の範囲判定: v_min ≤ V ≤ v_max
        nodes.append(helper.make_node("GreaterOrEqual", ["V", "v_min"], ["v_ge"]))
        nodes.append(helper.make_node("LessOrEqual", ["V", "v_max"], ["v_le"]))
        nodes.append(helper.make_node("And", ["v_ge", "v_le"], ["v_in"]))

        # H の範囲判定 (循環対応)
        # h_min <= h_max の場合: h_min ≤ H ≤ h_max
        # h_min > h_max の場合: H ≥ h_min OR H ≤ h_max (折り返し)
        nodes.append(helper.make_node("GreaterOrEqual", ["H", "h_min"], ["h_ge"]))
        nodes.append(helper.make_node("LessOrEqual", ["H", "h_max"], ["h_le"]))

        # 通常範囲: h_ge AND h_le
        nodes.append(helper.make_node("And", ["h_ge", "h_le"], ["h_normal"]))
        # 折り返し範囲: h_ge OR h_le
        nodes.append(helper.make_node("Or", ["h_ge", "h_le"], ["h_wrap"]))

        # h_min > h_max なら折り返しモード
        # Where(bool) が ONNX Runtime で未対応のため float 経由で選択
        nodes.append(helper.make_node("Greater", ["h_min", "h_max"], ["is_wrap"]))
        nodes.append(helper.make_node("Cast", ["h_normal"], ["h_normal_f"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Cast", ["h_wrap"], ["h_wrap_f"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Where", ["is_wrap", "h_wrap_f", "h_normal_f"], ["h_in_f"]))
        nodes.append(helper.make_node("Cast", ["h_in_f"], ["h_in"], to=TensorProto.BOOL))

        # 全チャネル AND
        nodes.append(helper.make_node("And", ["h_in", "s_in"], ["hs_in"]))
        nodes.append(helper.make_node("And", ["hs_in", "v_in"], ["mask"]))

        # bool → float → 3ch 拡張
        nodes.append(helper.make_node("Cast", ["mask"], ["mask_f"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Expand", ["mask_f", "expand_shape"], ["output"]))

        # --- 入出力定義 ---
        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        h_min_vi = helper.make_tensor_value_info("h_min", TensorProto.FLOAT, [1])
        h_max_vi = helper.make_tensor_value_info("h_max", TensorProto.FLOAT, [1])
        s_min_vi = helper.make_tensor_value_info("s_min", TensorProto.FLOAT, [1])
        s_max_vi = helper.make_tensor_value_info("s_max", TensorProto.FLOAT, [1])
        v_min_vi = helper.make_tensor_value_info("v_min", TensorProto.FLOAT, [1])
        v_max_vi = helper.make_tensor_value_info("v_max", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes,
            self.op_name,
            [input_vi, h_min_vi, h_max_vi, s_min_vi, s_max_vi, v_min_vi, v_max_vi],
            [output_vi],
            initializer=initializers,
        )
