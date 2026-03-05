"""任意領域クロップの ONNX グラフ定義.

正規化座標 (0〜1) で指定した領域を Slice で切り出す.
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class CropOp(OnnxGraphOp):
    """任意領域クロップ.

    正規化座標 crop_top, crop_left, crop_h, crop_w (各 0〜1) で
    切り出し領域を指定する。出力サイズは入力の H*crop_h × W*crop_w.
    """

    @property
    def op_name(self) -> str:
        return "crop"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("crop_top", TensorProto.FLOAT, [1]),
            ("crop_left", TensorProto.FLOAT, [1]),
            ("crop_h", TensorProto.FLOAT, [1]),
            ("crop_w", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H_out", "W_out"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {
            "crop_top": (0.0, 0.9, 0.0),
            "crop_left": (0.0, 0.9, 0.0),
            "crop_h": (0.1, 1.0, 1.0),
            "crop_w": (0.1, 1.0, 1.0),
        }

    def build_graph(self) -> GraphProto:
        nodes = []
        inits = []

        # Shape(input) → [N, 3, H, W]
        nodes.append(helper.make_node("Shape", ["input"], ["shape"]))

        # H, W を取得
        idx2 = numpy_helper.from_array(np.array(2, dtype=np.int64), "idx2")
        idx3 = numpy_helper.from_array(np.array(3, dtype=np.int64), "idx3")
        inits += [idx2, idx3]
        nodes.append(helper.make_node("Gather", ["shape", "idx2"], ["H_i64"], axis=0))
        nodes.append(helper.make_node("Gather", ["shape", "idx3"], ["W_i64"], axis=0))

        # int64 → float32
        nodes.append(helper.make_node("Cast", ["H_i64"], ["H_f"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Cast", ["W_i64"], ["W_f"], to=TensorProto.FLOAT))

        # ピクセル座標を計算
        # start_h = floor(crop_top * H)
        nodes.append(helper.make_node("Mul", ["crop_top", "H_f"], ["top_f"]))
        nodes.append(helper.make_node("Floor", ["top_f"], ["top_floor"]))
        nodes.append(helper.make_node("Cast", ["top_floor"], ["start_h"], to=TensorProto.INT64))

        # start_w = floor(crop_left * W)
        nodes.append(helper.make_node("Mul", ["crop_left", "W_f"], ["left_f"]))
        nodes.append(helper.make_node("Floor", ["left_f"], ["left_floor"]))
        nodes.append(helper.make_node("Cast", ["left_floor"], ["start_w"], to=TensorProto.INT64))

        # end_h = floor((crop_top + crop_h) * H), clamp to H
        nodes.append(helper.make_node("Add", ["crop_top", "crop_h"], ["bottom_ratio"]))
        nodes.append(helper.make_node("Mul", ["bottom_ratio", "H_f"], ["bottom_f"]))
        nodes.append(helper.make_node("Floor", ["bottom_f"], ["bottom_floor"]))
        nodes.append(helper.make_node("Cast", ["bottom_floor"], ["end_h_raw"], to=TensorProto.INT64))
        nodes.append(helper.make_node("Min", ["end_h_raw", "H_i64"], ["end_h"]))

        # end_w = floor((crop_left + crop_w) * W), clamp to W
        nodes.append(helper.make_node("Add", ["crop_left", "crop_w"], ["right_ratio"]))
        nodes.append(helper.make_node("Mul", ["right_ratio", "W_f"], ["right_f"]))
        nodes.append(helper.make_node("Floor", ["right_f"], ["right_floor"]))
        nodes.append(helper.make_node("Cast", ["right_floor"], ["end_w_raw"], to=TensorProto.INT64))
        nodes.append(helper.make_node("Min", ["end_w_raw", "W_i64"], ["end_w"]))

        # Slice(input, starts=[start_h, start_w], ends=[end_h, end_w], axes=[2, 3])
        # スカラー → 1D への変換 (Reshape)
        shape_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), "shape_1")
        inits.append(shape_1)

        nodes.append(helper.make_node("Reshape", ["start_h", "shape_1"], ["start_h_1d"]))
        nodes.append(helper.make_node("Reshape", ["start_w", "shape_1"], ["start_w_1d"]))
        nodes.append(helper.make_node("Reshape", ["end_h", "shape_1"], ["end_h_1d"]))
        nodes.append(helper.make_node("Reshape", ["end_w", "shape_1"], ["end_w_1d"]))

        nodes.append(helper.make_node(
            "Concat", ["start_h_1d", "start_w_1d"], ["starts"], axis=0
        ))
        nodes.append(helper.make_node(
            "Concat", ["end_h_1d", "end_w_1d"], ["ends"], axis=0
        ))

        axes_init = numpy_helper.from_array(
            np.array([2, 3], dtype=np.int64), "axes"
        )
        inits.append(axes_init)

        nodes.append(helper.make_node(
            "Slice", ["input", "starts", "ends", "axes"], ["output"]
        ))

        # 入出力定義
        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )
        crop_top_vi = helper.make_tensor_value_info(
            "crop_top", TensorProto.FLOAT, [1]
        )
        crop_left_vi = helper.make_tensor_value_info(
            "crop_left", TensorProto.FLOAT, [1]
        )
        crop_h_vi = helper.make_tensor_value_info(
            "crop_h", TensorProto.FLOAT, [1]
        )
        crop_w_vi = helper.make_tensor_value_info(
            "crop_w", TensorProto.FLOAT, [1]
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H_out", "W_out"]
        )

        return helper.make_graph(
            nodes,
            self.op_name,
            [input_vi, crop_top_vi, crop_left_vi, crop_h_vi, crop_w_vi],
            [output_vi],
            initializer=inits,
        )
