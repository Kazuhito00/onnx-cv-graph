"""HSV ベースの色抑制の ONNX グラフ定義.

指定した色相範囲・彩度範囲にマッチする画素を白に置換する.
赤スタンプ除去など OCR 前処理に有効.

パラメータ:
  h_center: 抑制対象の色相中心 [0, 1] (0=赤, 0.33=緑, 0.67=青)
  h_range:  色相の許容幅 [0, 0.5] (h_center ± h_range)
  s_min:    彩度の下限 [0, 1] (低彩度=無彩色を除外)
  strength: 抑制強度 [0, 1] (0=無効, 1=完全に白置換)

処理フロー:
  1. RGB→HSV 変換
  2. |H - h_center| < h_range AND S >= s_min → マスク生成
  3. output = input * (1 - mask*strength) + white * mask*strength
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class ColorSuppressOp(OnnxGraphOp):
    """HSV ベースの色抑制.

    RGB→HSV → 色相/彩度マスク → 白置換.
    """

    @property
    def op_name(self) -> str:
        return "color_suppress"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("h_center", TensorProto.FLOAT, [1]),
            ("h_range", TensorProto.FLOAT, [1]),
            ("s_min", TensorProto.FLOAT, [1]),
            ("strength", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {
            "h_center": (0.0, 1.0, 0.0),      # 0.0 = 赤
            "h_range": (0.0, 0.5, 0.08),       # ±0.08 ≈ ±29°
            "s_min": (0.0, 1.0, 0.3),          # 低彩度を除外
            "strength": (0.0, 1.0, 1.0),        # 完全置換
        }

    def build_graph(self) -> GraphProto:
        # --- 定数 ---
        eps = numpy_helper.from_array(np.array([1e-7], dtype=np.float32), name="eps")
        zero = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="zero")
        one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")
        sixty = numpy_helper.from_array(np.array([60.0], dtype=np.float32), name="sixty")
        three_sixty = numpy_helper.from_array(np.array([360.0], dtype=np.float32), name="three_sixty")
        two = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name="two")
        four = numpy_helper.from_array(np.array([4.0], dtype=np.float32), name="four")

        idx_r = numpy_helper.from_array(np.array([0], dtype=np.int64), name="idx_r")
        idx_g = numpy_helper.from_array(np.array([1], dtype=np.int64), name="idx_g")
        idx_b = numpy_helper.from_array(np.array([2], dtype=np.int64), name="idx_b")
        expand_shape = numpy_helper.from_array(np.array([1, 3, 1, 1], dtype=np.int64), name="expand_shape")

        initializers = [
            eps, zero, one, sixty, three_sixty, two, four,
            idx_r, idx_g, idx_b, expand_shape,
        ]

        nodes = []

        # ===== RGB → HSV 変換 =====
        nodes.append(helper.make_node("Gather", ["input", "idx_r"], ["R"], axis=1))
        nodes.append(helper.make_node("Gather", ["input", "idx_g"], ["G"], axis=1))
        nodes.append(helper.make_node("Gather", ["input", "idx_b"], ["B"], axis=1))

        nodes.append(helper.make_node("Max", ["R", "G", "B"], ["V"]))
        nodes.append(helper.make_node("Min", ["R", "G", "B"], ["Vmin"]))
        nodes.append(helper.make_node("Sub", ["V", "Vmin"], ["diff"]))

        nodes.append(helper.make_node("Add", ["V", "eps"], ["V_safe"]))
        nodes.append(helper.make_node("Div", ["diff", "V_safe"], ["S"]))

        nodes.append(helper.make_node("Add", ["diff", "eps"], ["diff_safe"]))

        # H 計算
        nodes.append(helper.make_node("Sub", ["G", "B"], ["gb"]))
        nodes.append(helper.make_node("Div", ["gb", "diff_safe"], ["h_r"]))

        nodes.append(helper.make_node("Sub", ["B", "R"], ["br"]))
        nodes.append(helper.make_node("Div", ["br", "diff_safe"], ["br_norm"]))
        nodes.append(helper.make_node("Add", ["two", "br_norm"], ["h_g"]))

        nodes.append(helper.make_node("Sub", ["R", "G"], ["rg"]))
        nodes.append(helper.make_node("Div", ["rg", "diff_safe"], ["rg_norm"]))
        nodes.append(helper.make_node("Add", ["four", "rg_norm"], ["h_b"]))

        nodes.append(helper.make_node("Equal", ["V", "B"], ["is_b"]))
        nodes.append(helper.make_node("Where", ["is_b", "h_b", "h_r"], ["h_tmp1"]))
        nodes.append(helper.make_node("Equal", ["V", "G"], ["is_g"]))
        nodes.append(helper.make_node("Where", ["is_g", "h_g", "h_tmp1"], ["h_raw"]))

        nodes.append(helper.make_node("Mul", ["h_raw", "sixty"], ["h_deg"]))
        nodes.append(helper.make_node("Less", ["h_deg", "zero"], ["h_neg"]))
        nodes.append(helper.make_node("Add", ["h_deg", "three_sixty"], ["h_plus360"]))
        nodes.append(helper.make_node("Where", ["h_neg", "h_plus360", "h_deg"], ["h_deg_pos"]))
        nodes.append(helper.make_node("Div", ["h_deg_pos", "three_sixty"], ["H_raw"]))

        nodes.append(helper.make_node("Equal", ["diff", "zero"], ["is_gray"]))
        nodes.append(helper.make_node("Where", ["is_gray", "zero", "H_raw"], ["H"]))

        # ===== 色相距離の計算 (循環対応) =====
        # h_diff = |H - h_center|
        nodes.append(helper.make_node("Sub", ["H", "h_center"], ["h_diff_raw"]))
        nodes.append(helper.make_node("Abs", ["h_diff_raw"], ["h_diff_abs"]))
        # 循環距離: min(h_diff_abs, 1.0 - h_diff_abs)
        nodes.append(helper.make_node("Sub", ["one", "h_diff_abs"], ["h_diff_wrap"]))
        nodes.append(helper.make_node("Min", ["h_diff_abs", "h_diff_wrap"], ["h_dist"]))

        # ===== マスク生成 =====
        # 色相条件: h_dist < h_range
        nodes.append(helper.make_node("Less", ["h_dist", "h_range"], ["h_match"]))
        # 彩度条件: S >= s_min
        nodes.append(helper.make_node("GreaterOrEqual", ["S", "s_min"], ["s_match"]))
        # AND
        nodes.append(helper.make_node("And", ["h_match", "s_match"], ["mask_bool"]))

        # mask_bool → float (N, 1, H, W) → (N, 3, H, W)
        nodes.append(helper.make_node("Cast", ["mask_bool"], ["mask_1ch"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Expand", ["mask_1ch", "expand_shape"], ["mask_3ch"]))

        # ===== 白置換 =====
        # blend_weight = mask_3ch * strength
        nodes.append(helper.make_node("Mul", ["mask_3ch", "strength"], ["blend_w"]))
        # output = input * (1 - blend_w) + 1.0 * blend_w
        nodes.append(helper.make_node("Sub", ["one", "blend_w"], ["inv_w"]))
        nodes.append(helper.make_node("Mul", ["input", "inv_w"], ["keep_part"]))
        nodes.append(helper.make_node("Add", ["keep_part", "blend_w"], ["output"]))

        # --- 入出力定義 ---
        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        hc_vi = helper.make_tensor_value_info("h_center", TensorProto.FLOAT, [1])
        hr_vi = helper.make_tensor_value_info("h_range", TensorProto.FLOAT, [1])
        sm_vi = helper.make_tensor_value_info("s_min", TensorProto.FLOAT, [1])
        st_vi = helper.make_tensor_value_info("strength", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name,
            [input_vi, hc_vi, hr_vi, sm_vi, st_vi], [output_vi],
            initializer=initializers,
        )
