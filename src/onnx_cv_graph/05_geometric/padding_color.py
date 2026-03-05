"""任意色パディングの ONNX グラフ定義.

OpenCV の copyMakeBorder(BORDER_CONSTANT) 相当.
画像の上下左右に指定比率のパディングを追加し、指定 RGB 色で塗り潰す.

方式:
  1. Pad(input, constant=0) で拡張
  2. ConstantOfShape(input_shape, 1.0) + Pad(constant=0) でマスク生成
  3. color [1,3,1,1] * (1 - mask) でパディング領域の色塗り
  4. output = padded + color_fill
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class PaddingColorOp(OnnxGraphOp):
    """任意色パディング (constant モード).

    pad_ratio で上下左右のパディング幅を比率指定し、
    pad_r / pad_g / pad_b でパディング色を [0, 1] で指定する。
    """

    @property
    def op_name(self) -> str:
        return "padding_color"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("pad_ratio", TensorProto.FLOAT, [1]),
            ("pad_r", TensorProto.FLOAT, [1]),
            ("pad_g", TensorProto.FLOAT, [1]),
            ("pad_b", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N_out", 3, "H_out", "W_out"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {
            "pad_ratio": (0.0, 0.5, 0.1),
            "pad_r": (0.0, 1.0, 0.0),
            "pad_g": (0.0, 1.0, 0.0),
            "pad_b": (0.0, 1.0, 0.0),
        }

    def build_graph(self) -> GraphProto:
        nodes = []
        inits = []

        # --- 定数 ---
        zero_pads = numpy_helper.from_array(
            np.array([0, 0], dtype=np.int64), "zero_pads",
        )
        h_idx = numpy_helper.from_array(np.array([2], dtype=np.int64), "h_idx")
        w_idx = numpy_helper.from_array(np.array([3], dtype=np.int64), "w_idx")
        zero_f = numpy_helper.from_array(np.float32(0.0), "zero_f")
        one_val = numpy_helper.from_array(
            np.array([1.0], dtype=np.float32), "one_val",
        )
        color_shape = numpy_helper.from_array(
            np.array([1, 3, 1, 1], dtype=np.int64), "color_shape",
        )
        inits += [zero_pads, h_idx, w_idx, zero_f, one_val, color_shape]

        # --- パディング幅の計算 ---
        nodes.append(helper.make_node("Shape", ["input"], ["shape"]))
        nodes.append(helper.make_node("Gather", ["shape", "h_idx"], ["H_i64"]))
        nodes.append(helper.make_node("Gather", ["shape", "w_idx"], ["W_i64"]))

        nodes.append(helper.make_node("Cast", ["H_i64"], ["H_f"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Cast", ["W_i64"], ["W_f"], to=TensorProto.FLOAT))

        nodes.append(helper.make_node("Mul", ["H_f", "pad_ratio"], ["pad_h_f"]))
        nodes.append(helper.make_node("Floor", ["pad_h_f"], ["pad_h_floor"]))
        nodes.append(helper.make_node("Cast", ["pad_h_floor"], ["pad_h"], to=TensorProto.INT64))

        nodes.append(helper.make_node("Mul", ["W_f", "pad_ratio"], ["pad_w_f"]))
        nodes.append(helper.make_node("Floor", ["pad_w_f"], ["pad_w_floor"]))
        nodes.append(helper.make_node("Cast", ["pad_w_floor"], ["pad_w"], to=TensorProto.INT64))

        # pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]
        nodes.append(helper.make_node(
            "Concat",
            ["zero_pads", "pad_h", "pad_w", "zero_pads", "pad_h", "pad_w"],
            ["pads"], axis=0,
        ))

        # --- 入力画像をゼロパディング ---
        nodes.append(helper.make_node(
            "Pad", ["input", "pads", "zero_f"], ["padded"], mode="constant",
        ))

        # --- マスク生成: 元画像領域=1, パディング領域=0 ---
        nodes.append(helper.make_node("Shape", ["input"], ["in_shape"]))
        nodes.append(helper.make_node(
            "ConstantOfShape", ["in_shape"], ["ones"],
            value=helper.make_tensor("val", TensorProto.FLOAT, [1], [1.0]),
        ))
        nodes.append(helper.make_node(
            "Pad", ["ones", "pads", "zero_f"], ["mask"], mode="constant",
        ))

        # --- パディング色テンソル [1, 3, 1, 1] ---
        nodes.append(helper.make_node(
            "Concat", ["pad_r", "pad_g", "pad_b"], ["color_flat"], axis=0,
        ))
        nodes.append(helper.make_node(
            "Reshape", ["color_flat", "color_shape"], ["color"],
        ))

        # --- 合成: output = padded + color * (1 - mask) ---
        nodes.append(helper.make_node("Sub", ["one_val", "mask"], ["inv_mask"]))
        nodes.append(helper.make_node("Mul", ["color", "inv_mask"], ["color_fill"]))
        nodes.append(helper.make_node("Add", ["padded", "color_fill"], ["output"]))

        # --- 入出力定義 ---
        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", 3, "H", "W"],
        )
        ratio_vi = helper.make_tensor_value_info("pad_ratio", TensorProto.FLOAT, [1])
        r_vi = helper.make_tensor_value_info("pad_r", TensorProto.FLOAT, [1])
        g_vi = helper.make_tensor_value_info("pad_g", TensorProto.FLOAT, [1])
        b_vi = helper.make_tensor_value_info("pad_b", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N_out", 3, "H_out", "W_out"],
        )

        return helper.make_graph(
            nodes, self.op_name,
            [input_vi, ratio_vi, r_vi, g_vi, b_vi], [output_vi],
            initializer=inits,
        )
