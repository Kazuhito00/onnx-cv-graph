"""レベル補正の ONNX グラフ定義."""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class LevelsOp(OnnxGraphOp):
    """レベル補正.

    Photoshop のレベル補正相当の操作.
    入力レンジ (in_black, in_white)、ガンマ、出力レンジ (out_black, out_white) の
    5パラメータで明暗を調整する.
    全パラメータがデフォルト値の場合は恒等変換となる.

    処理フロー:
    1. Clip(in_black, in_white) で入力レンジにクランプ
    2. Sub(in_black) でゼロ基点にシフト
    3. Div(in_white - in_black) で [0,1] に正規化
    4. Pow(gamma) でガンマ補正
    5. Mul(out_white - out_black) + Add(out_black) で出力レンジにマッピング
    6. Clip(0, 1) で画像ドメイン保証
    """

    @property
    def op_name(self) -> str:
        return "levels"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("in_black", TensorProto.FLOAT, [1]),
            ("in_white", TensorProto.FLOAT, [1]),
            ("gamma", TensorProto.FLOAT, [1]),
            ("out_black", TensorProto.FLOAT, [1]),
            ("out_white", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {
            "in_black": (0.0, 1.0, 0.0),
            "in_white": (0.0, 1.0, 1.0),
            "gamma": (0.2, 5.0, 1.0),
            "out_black": (0.0, 1.0, 0.0),
            "out_white": (0.0, 1.0, 1.0),
        }

    def build_graph(self) -> GraphProto:
        # 定数
        clip_min = numpy_helper.from_array(
            np.array(0.0, dtype=np.float32), name="clip_min"
        )
        clip_max = numpy_helper.from_array(
            np.array(1.0, dtype=np.float32), name="clip_max"
        )
        epsilon = numpy_helper.from_array(
            np.array(1e-6, dtype=np.float32), name="epsilon"
        )

        # 1. Clip(in_black, in_white): 入力レンジにクランプ
        clip_input = helper.make_node(
            "Clip", inputs=["input", "in_black", "in_white"], outputs=["clamped"]
        )

        # 2. Sub(in_black): ゼロ基点にシフト
        sub_black = helper.make_node(
            "Sub", inputs=["clamped", "in_black"], outputs=["shifted"]
        )

        # 3. レンジ幅を計算: in_white - in_black
        sub_range = helper.make_node(
            "Sub", inputs=["in_white", "in_black"], outputs=["range_raw"]
        )

        # ゼロ除算防止: Max(range, ε)
        max_eps = helper.make_node(
            "Max", inputs=["range_raw", "epsilon"], outputs=["range_safe"]
        )

        # 4. Div: [0,1] に正規化
        div_range = helper.make_node(
            "Div", inputs=["shifted", "range_safe"], outputs=["normalized"]
        )

        # 5. Clip(0, 1): Pow 前の安全保証
        clip_pre_pow = helper.make_node(
            "Clip", inputs=["normalized", "clip_min", "clip_max"], outputs=["safe"]
        )

        # 6. Pow(gamma): ガンマ補正
        pow_node = helper.make_node(
            "Pow", inputs=["safe", "gamma"], outputs=["gamma_corrected"]
        )

        # 7. 出力レンジ幅: out_white - out_black
        sub_out_range = helper.make_node(
            "Sub", inputs=["out_white", "out_black"], outputs=["out_range"]
        )

        # 8. Mul(out_range): 出力レンジ幅にスケーリング
        mul_out = helper.make_node(
            "Mul", inputs=["gamma_corrected", "out_range"], outputs=["scaled"]
        )

        # 9. Add(out_black): 出力黒点にシフト
        add_out = helper.make_node(
            "Add", inputs=["scaled", "out_black"], outputs=["mapped"]
        )

        # 10. Clip(0, 1): 画像ドメイン [0,1] 保証
        clip_final = helper.make_node(
            "Clip", inputs=["mapped", "clip_min", "clip_max"], outputs=["output"]
        )

        # value_info 定義
        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )
        in_black_vi = helper.make_tensor_value_info(
            "in_black", TensorProto.FLOAT, [1]
        )
        in_white_vi = helper.make_tensor_value_info(
            "in_white", TensorProto.FLOAT, [1]
        )
        gamma_vi = helper.make_tensor_value_info(
            "gamma", TensorProto.FLOAT, [1]
        )
        out_black_vi = helper.make_tensor_value_info(
            "out_black", TensorProto.FLOAT, [1]
        )
        out_white_vi = helper.make_tensor_value_info(
            "out_white", TensorProto.FLOAT, [1]
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )

        graph = helper.make_graph(
            [clip_input, sub_black, sub_range, max_eps, div_range,
             clip_pre_pow, pow_node, sub_out_range, mul_out, add_out, clip_final],
            self.op_name,
            [input_vi, in_black_vi, in_white_vi, gamma_vi,
             out_black_vi, out_white_vi],
            [output_vi],
            initializer=[clip_min, clip_max, epsilon],
        )
        return graph
