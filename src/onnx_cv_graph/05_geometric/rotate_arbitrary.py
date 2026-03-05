"""任意角度回転の ONNX グラフ定義.

GridSample ベースで任意角度の回転を実現する。
度 → ラジアン → Cos/Sin → 回転行列適用 → GridSample(bilinear) → Clip(0,1)。
出力サイズは入力と同一 (はみ出し部分は zeros パディング)。
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec

from ._grid_utils import build_gridsample_nodes, build_meshgrid_nodes


class RotateArbitraryOp(OnnxGraphOp):
    """任意角度回転 (GridSample ベース).

    角度パラメータ (度) で指定した回転を適用する。
    出力サイズは入力と同一で、はみ出し部分はゼロパディングされる。
    """

    @property
    def op_name(self) -> str:
        return "rotate_arbitrary"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("angle", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"angle": (-180.0, 180.0, 0.0)}

    def build_graph(self) -> GraphProto:
        nodes = []
        inits = []

        # --- メッシュグリッド構築 ---
        mg_nodes, mg_inits, gx, gy = build_meshgrid_nodes("mg")
        nodes += mg_nodes
        inits += mg_inits

        # --- 度 → ラジアン ---
        deg2rad = numpy_helper.from_array(
            np.float32(np.pi / 180.0), "deg2rad"
        )
        inits.append(deg2rad)
        nodes.append(helper.make_node("Mul", ["angle", "deg2rad"], ["angle_rad"]))

        # --- Cos / Sin ---
        nodes.append(helper.make_node("Cos", ["angle_rad"], ["cos_a"]))
        nodes.append(helper.make_node("Sin", ["angle_rad"], ["sin_a"]))

        # --- 回転行列適用 (逆マッピング) ---
        # x_src =  cos*x + sin*y
        # y_src = -sin*x + cos*y
        nodes.append(helper.make_node("Mul", ["cos_a", gx], ["cos_x"]))
        nodes.append(helper.make_node("Mul", ["sin_a", gy], ["sin_y"]))
        nodes.append(helper.make_node("Add", ["cos_x", "sin_y"], ["x_src"]))

        nodes.append(helper.make_node("Mul", ["sin_a", gx], ["sin_x"]))
        nodes.append(helper.make_node("Neg", ["sin_x"], ["neg_sin_x"]))
        nodes.append(helper.make_node("Mul", ["cos_a", gy], ["cos_y"]))
        nodes.append(helper.make_node("Add", ["neg_sin_x", "cos_y"], ["y_src"]))

        # --- GridSample + Clip ---
        gs_nodes, gs_inits = build_gridsample_nodes("x_src", "y_src")
        nodes += gs_nodes
        inits += gs_inits

        # --- 入出力定義 ---
        input_vi = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )
        angle_vi = helper.make_tensor_value_info(
            "angle", TensorProto.FLOAT, [1]
        )
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )

        return helper.make_graph(
            nodes, self.op_name, [input_vi, angle_vi], [output_vi],
            initializer=inits,
        )
