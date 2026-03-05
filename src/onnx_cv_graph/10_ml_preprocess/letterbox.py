"""Letterbox リサイズの ONNX グラフ定義.

アスペクト比を維持しつつ target_h × target_w にリサイズし、
余白をグレー (0.5) でパディングする. YOLO 等の物体検出前処理で使用.
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class LetterboxOp(OnnxGraphOp):
    """Letterbox リサイズ.

    アスペクト比を維持して target_h × target_w にリサイズし、
    余白を 0.5 (グレー) でパディングする.
    """

    @property
    def op_name(self) -> str:
        return "letterbox"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("target_h", TensorProto.FLOAT, [1]),
            ("target_w", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H_out", "W_out"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {
            "target_h": (64.0, 1024.0, 640.0),
            "target_w": (64.0, 1024.0, 640.0),
        }

    def build_graph(self) -> GraphProto:
        nodes = []
        inits = []

        # --- 入力画像の H, W を取得 ---
        nodes.append(helper.make_node("Shape", ["input"], ["in_shape"]))
        idx2 = numpy_helper.from_array(np.array(2, dtype=np.int64), "idx2")
        idx3 = numpy_helper.from_array(np.array(3, dtype=np.int64), "idx3")
        inits += [idx2, idx3]
        nodes.append(helper.make_node("Gather", ["in_shape", "idx2"], ["H_i64"], axis=0))
        nodes.append(helper.make_node("Gather", ["in_shape", "idx3"], ["W_i64"], axis=0))
        nodes.append(helper.make_node("Cast", ["H_i64"], ["H_f"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Cast", ["W_i64"], ["W_f"], to=TensorProto.FLOAT))

        # --- スケール計算: scale = min(target_h/H, target_w/W) ---
        nodes.append(helper.make_node("Div", ["target_h", "H_f"], ["scale_h"]))
        nodes.append(helper.make_node("Div", ["target_w", "W_f"], ["scale_w"]))
        nodes.append(helper.make_node("Min", ["scale_h", "scale_w"], ["scale"]))

        # --- Resize: scales = [1, 1, scale, scale] ---
        one_f = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one_f")
        inits.append(one_f)
        nodes.append(helper.make_node(
            "Concat", ["one_f", "one_f", "scale", "scale"], ["scales"], axis=0
        ))

        roi = numpy_helper.from_array(np.array([], dtype=np.float32), "roi")
        inits.append(roi)
        nodes.append(helper.make_node(
            "Resize",
            inputs=["input", "roi", "scales"],
            outputs=["resized"],
            mode="linear",
        ))

        # --- リサイズ後の H, W を取得 ---
        nodes.append(helper.make_node("Shape", ["resized"], ["res_shape"]))
        nodes.append(helper.make_node("Gather", ["res_shape", "idx2"], ["newH_i64"], axis=0))
        nodes.append(helper.make_node("Gather", ["res_shape", "idx3"], ["newW_i64"], axis=0))
        nodes.append(helper.make_node("Cast", ["newH_i64"], ["newH_f"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Cast", ["newW_i64"], ["newW_f"], to=TensorProto.FLOAT))

        # --- パディング量計算 ---
        nodes.append(helper.make_node("Sub", ["target_h", "newH_f"], ["pad_h_total"]))
        nodes.append(helper.make_node("Sub", ["target_w", "newW_f"], ["pad_w_total"]))

        two = numpy_helper.from_array(np.array([2.0], dtype=np.float32), "two")
        inits.append(two)
        nodes.append(helper.make_node("Div", ["pad_h_total", "two"], ["pad_h_half"]))
        nodes.append(helper.make_node("Floor", ["pad_h_half"], ["pad_top_f"]))
        nodes.append(helper.make_node("Sub", ["pad_h_total", "pad_top_f"], ["pad_bottom_f"]))

        nodes.append(helper.make_node("Div", ["pad_w_total", "two"], ["pad_w_half"]))
        nodes.append(helper.make_node("Floor", ["pad_w_half"], ["pad_left_f"]))
        nodes.append(helper.make_node("Sub", ["pad_w_total", "pad_left_f"], ["pad_right_f"]))

        # float → int64 にキャスト
        nodes.append(helper.make_node("Cast", ["pad_top_f"], ["pad_top"], to=TensorProto.INT64))
        nodes.append(helper.make_node("Cast", ["pad_bottom_f"], ["pad_bottom"], to=TensorProto.INT64))
        nodes.append(helper.make_node("Cast", ["pad_left_f"], ["pad_left"], to=TensorProto.INT64))
        nodes.append(helper.make_node("Cast", ["pad_right_f"], ["pad_right"], to=TensorProto.INT64))

        # --- pads = [0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right] ---
        shape_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), "shape_1")
        zero_1d = numpy_helper.from_array(np.array([0], dtype=np.int64), "zero_1d")
        inits += [shape_1, zero_1d]

        nodes.append(helper.make_node("Reshape", ["pad_top", "shape_1"], ["pad_top_1d"]))
        nodes.append(helper.make_node("Reshape", ["pad_bottom", "shape_1"], ["pad_bottom_1d"]))
        nodes.append(helper.make_node("Reshape", ["pad_left", "shape_1"], ["pad_left_1d"]))
        nodes.append(helper.make_node("Reshape", ["pad_right", "shape_1"], ["pad_right_1d"]))

        # NCHW: [N_begin, C_begin, H_begin, W_begin, N_end, C_end, H_end, W_end]
        nodes.append(helper.make_node(
            "Concat",
            ["zero_1d", "zero_1d", "pad_top_1d", "pad_left_1d",
             "zero_1d", "zero_1d", "pad_bottom_1d", "pad_right_1d"],
            ["pads"], axis=0
        ))

        # --- Pad (constant, value=0.5) ---
        pad_val = numpy_helper.from_array(np.array([0.5], dtype=np.float32), "pad_val")
        inits.append(pad_val)
        nodes.append(helper.make_node(
            "Pad", ["resized", "pads", "pad_val"], ["output"], mode="constant"
        ))

        # 入出力定義
        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )
        target_h_vi = helper.make_tensor_value_info(
            "target_h", TensorProto.FLOAT, [1]
        )
        target_w_vi = helper.make_tensor_value_info(
            "target_w", TensorProto.FLOAT, [1]
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H_out", "W_out"]
        )

        return helper.make_graph(
            nodes,
            self.op_name,
            [input_vi, target_h_vi, target_w_vi],
            [output_vi],
            initializer=inits,
        )
