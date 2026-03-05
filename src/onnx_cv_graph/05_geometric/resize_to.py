"""任意サイズリサイズの ONNX グラフ定義.

target_h, target_w を推論時パラメータとして受け取り、
ONNX Resize オペレータで双線形補間リサイズを実現する.
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class ResizeToOp(OnnxGraphOp):
    """任意サイズリサイズ (双線形補間).

    入力 (N,3,H,W) float32 に対し、target_h × target_w にリサイズする.
    target_h, target_w は推論時に指定する.
    """

    @property
    def op_name(self) -> str:
        return "resize_to"

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
            "target_h": (1.0, 4096.0, 256.0),
            "target_w": (1.0, 4096.0, 256.0),
        }

    def build_graph(self) -> GraphProto:
        # target_h, target_w を int64 にキャストして sizes = [N, 3, target_h, target_w] を構築
        # Resize op は sizes 入力で絶対サイズ指定が可能

        # N と C は入力から Shape で取得
        shape_node = helper.make_node("Shape", ["input"], ["input_shape"])
        # input_shape[0:2] → [N, 3]
        nc_start = numpy_helper.from_array(np.array([0], dtype=np.int64), "nc_start")
        nc_end = numpy_helper.from_array(np.array([2], dtype=np.int64), "nc_end")
        slice_nc = helper.make_node("Slice", ["input_shape", "nc_start", "nc_end"], ["nc_dims"])

        # target_h, target_w を int64 にキャスト
        cast_h = helper.make_node("Cast", ["target_h"], ["target_h_i64"], to=TensorProto.INT64)
        cast_w = helper.make_node("Cast", ["target_w"], ["target_w_i64"], to=TensorProto.INT64)

        # [target_h, target_w] を連結
        concat_hw = helper.make_node("Concat", ["target_h_i64", "target_w_i64"], ["hw_dims"], axis=0)

        # sizes = [N, 3, target_h, target_w]
        concat_sizes = helper.make_node("Concat", ["nc_dims", "hw_dims"], ["sizes"], axis=0)

        # Resize (sizes 指定)
        roi = numpy_helper.from_array(np.array([], dtype=np.float32), "roi")
        scales = numpy_helper.from_array(np.array([], dtype=np.float32), "scales_empty")
        resize_node = helper.make_node(
            "Resize",
            inputs=["input", "roi", "scales_empty", "sizes"],
            outputs=["output"],
            mode="linear",
        )

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        target_h_vi = helper.make_tensor_value_info("target_h", TensorProto.FLOAT, [1])
        target_w_vi = helper.make_tensor_value_info("target_w", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H_out", "W_out"])

        return helper.make_graph(
            [shape_node, slice_nc, cast_h, cast_w, concat_hw, concat_sizes, resize_node],
            self.op_name,
            [input_vi, target_h_vi, target_w_vi],
            [output_vi],
            initializer=[nc_start, nc_end, roi, scales],
        )
