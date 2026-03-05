"""センタークロップの ONNX グラフ定義.

crop_ratio で指定した比率で中央を切り出す.
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class CenterCropOp(OnnxGraphOp):
    """センタークロップ.

    crop_ratio (0.1〜1.0) で入力の中央を切り出す.
    出力サイズは floor(H * ratio) × floor(W * ratio).
    """

    @property
    def op_name(self) -> str:
        return "center_crop"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("crop_ratio", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H_out", "W_out"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {
            "crop_ratio": (0.1, 1.0, 0.5),
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

        # crop_h = floor(H * ratio), crop_w = floor(W * ratio)
        nodes.append(helper.make_node("Mul", ["H_f", "crop_ratio"], ["crop_h_f"]))
        nodes.append(helper.make_node("Floor", ["crop_h_f"], ["crop_h_floor"]))
        nodes.append(helper.make_node("Mul", ["W_f", "crop_ratio"], ["crop_w_f"]))
        nodes.append(helper.make_node("Floor", ["crop_w_f"], ["crop_w_floor"]))

        # off_h = floor((H - crop_h) / 2), off_w = floor((W - crop_w) / 2)
        two = numpy_helper.from_array(np.array([2.0], dtype=np.float32), "two")
        inits.append(two)
        nodes.append(helper.make_node("Sub", ["H_f", "crop_h_floor"], ["rem_h"]))
        nodes.append(helper.make_node("Div", ["rem_h", "two"], ["off_h_f"]))
        nodes.append(helper.make_node("Floor", ["off_h_f"], ["off_h_floor"]))
        nodes.append(helper.make_node("Cast", ["off_h_floor"], ["start_h"], to=TensorProto.INT64))

        nodes.append(helper.make_node("Sub", ["W_f", "crop_w_floor"], ["rem_w"]))
        nodes.append(helper.make_node("Div", ["rem_w", "two"], ["off_w_f"]))
        nodes.append(helper.make_node("Floor", ["off_w_f"], ["off_w_floor"]))
        nodes.append(helper.make_node("Cast", ["off_w_floor"], ["start_w"], to=TensorProto.INT64))

        # end_h = start_h + crop_h, end_w = start_w + crop_w
        nodes.append(helper.make_node("Cast", ["crop_h_floor"], ["crop_h_i64"], to=TensorProto.INT64))
        nodes.append(helper.make_node("Cast", ["crop_w_floor"], ["crop_w_i64"], to=TensorProto.INT64))
        nodes.append(helper.make_node("Add", ["start_h", "crop_h_i64"], ["end_h"]))
        nodes.append(helper.make_node("Add", ["start_w", "crop_w_i64"], ["end_w"]))

        # Slice(input, starts, ends, axes=[2,3])
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
        crop_ratio_vi = helper.make_tensor_value_info(
            "crop_ratio", TensorProto.FLOAT, [1]
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H_out", "W_out"]
        )

        return helper.make_graph(
            nodes,
            self.op_name,
            [input_vi, crop_ratio_vi],
            [output_vi],
            initializer=inits,
        )
