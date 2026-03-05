"""射影変換 (ホモグラフィ) の ONNX グラフ定義.

GridSample ベースで射影変換を実現する。
8 パラメータ (p00..p21) によるホモグラフィ逆マッピング:
  w     = p20*x + p21*y + 1
  x_src = (p00*x + p01*y + p02) / w
  y_src = (p10*x + p11*y + p12) / w
ゼロ除算防止: w_safe = Sign(w) * Max(Abs(w), 1e-7)。
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec

from ._grid_utils import build_gridsample_nodes, build_meshgrid_nodes


class PerspectiveOp(OnnxGraphOp):
    """射影変換 (GridSample ベース).

    8 パラメータでホモグラフィ変換を適用する。
    p20, p21 が射影成分で、0 のときアフィン変換に退化する。
    """

    @property
    def op_name(self) -> str:
        return "perspective"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("p00", TensorProto.FLOAT, [1]),
            ("p01", TensorProto.FLOAT, [1]),
            ("p02", TensorProto.FLOAT, [1]),
            ("p10", TensorProto.FLOAT, [1]),
            ("p11", TensorProto.FLOAT, [1]),
            ("p12", TensorProto.FLOAT, [1]),
            ("p20", TensorProto.FLOAT, [1]),
            ("p21", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {
            "p00": (-2.0, 2.0, 1.0),
            "p01": (-2.0, 2.0, 0.0),
            "p02": (-1.0, 1.0, 0.0),
            "p10": (-2.0, 2.0, 0.0),
            "p11": (-2.0, 2.0, 1.0),
            "p12": (-1.0, 1.0, 0.0),
            "p20": (-1.0, 1.0, 0.0),
            "p21": (-1.0, 1.0, 0.0),
        }

    def build_graph(self) -> GraphProto:
        nodes = []
        inits = []

        # --- メッシュグリッド構築 ---
        mg_nodes, mg_inits, gx, gy = build_meshgrid_nodes("mg")
        nodes += mg_nodes
        inits += mg_inits

        # --- ホモグラフィ逆マッピング ---
        # w = p20*x + p21*y + 1
        one_f = numpy_helper.from_array(np.float32(1.0), "one_f")
        eps = numpy_helper.from_array(np.float32(1e-7), "eps")
        inits += [one_f, eps]

        nodes.append(helper.make_node("Mul", ["p20", gx], ["p20x"]))
        nodes.append(helper.make_node("Mul", ["p21", gy], ["p21y"]))
        nodes.append(helper.make_node("Add", ["p20x", "p21y"], ["p20x_p21y"]))
        nodes.append(helper.make_node("Add", ["p20x_p21y", "one_f"], ["w"]))

        # ゼロ除算防止: w_safe = Sign(w) * Max(Abs(w), eps)
        nodes.append(helper.make_node("Sign", ["w"], ["w_sign"]))
        nodes.append(helper.make_node("Abs", ["w"], ["w_abs"]))
        nodes.append(helper.make_node("Max", ["w_abs", "eps"], ["w_clamped"]))
        nodes.append(helper.make_node("Mul", ["w_sign", "w_clamped"], ["w_safe"]))

        # x_num = p00*x + p01*y + p02
        nodes.append(helper.make_node("Mul", ["p00", gx], ["p00x"]))
        nodes.append(helper.make_node("Mul", ["p01", gy], ["p01y"]))
        nodes.append(helper.make_node("Add", ["p00x", "p01y"], ["p00x_p01y"]))
        nodes.append(helper.make_node("Add", ["p00x_p01y", "p02"], ["x_num"]))

        # y_num = p10*x + p11*y + p12
        nodes.append(helper.make_node("Mul", ["p10", gx], ["p10x"]))
        nodes.append(helper.make_node("Mul", ["p11", gy], ["p11y"]))
        nodes.append(helper.make_node("Add", ["p10x", "p11y"], ["p10x_p11y"]))
        nodes.append(helper.make_node("Add", ["p10x_p11y", "p12"], ["y_num"]))

        # x_src = x_num / w_safe, y_src = y_num / w_safe
        nodes.append(helper.make_node("Div", ["x_num", "w_safe"], ["x_src"]))
        nodes.append(helper.make_node("Div", ["y_num", "w_safe"], ["y_src"]))

        # --- GridSample + Clip ---
        gs_nodes, gs_inits = build_gridsample_nodes("x_src", "y_src")
        nodes += gs_nodes
        inits += gs_inits

        # --- 入出力定義 ---
        inputs = [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            helper.make_tensor_value_info("p00", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("p01", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("p02", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("p10", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("p11", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("p12", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("p20", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("p21", TensorProto.FLOAT, [1]),
        ]
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )

        return helper.make_graph(
            nodes, self.op_name, inputs, [output_vi],
            initializer=inits,
        )
