"""アフィン変換の ONNX グラフ定義.

GridSample ベースで 2×3 アフィン変換を実現する。
6 パラメータ (a, b, tx, c, d, ty) による逆マッピング:
  x_src = a*x + b*y + tx
  y_src = c*x + d*y + ty
座標は [-1, 1] 正規化。identity: a=1, b=0, tx=0, c=0, d=1, ty=0。
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec

from ._grid_utils import build_gridsample_nodes, build_meshgrid_nodes


class AffineOp(OnnxGraphOp):
    """アフィン変換 (GridSample ベース).

    6 パラメータで任意のアフィン変換 (回転・スケーリング・せん断・平行移動) を適用する。
    座標は [-1, 1] 正規化空間で指定する。
    """

    @property
    def op_name(self) -> str:
        return "affine"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("a", TensorProto.FLOAT, [1]),
            ("b", TensorProto.FLOAT, [1]),
            ("tx", TensorProto.FLOAT, [1]),
            ("c", TensorProto.FLOAT, [1]),
            ("d", TensorProto.FLOAT, [1]),
            ("ty", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {
            "a": (-2.0, 2.0, 1.0),
            "b": (-2.0, 2.0, 0.0),
            "tx": (-1.0, 1.0, 0.0),
            "c": (-2.0, 2.0, 0.0),
            "d": (-2.0, 2.0, 1.0),
            "ty": (-1.0, 1.0, 0.0),
        }

    def build_graph(self) -> GraphProto:
        nodes = []
        inits = []

        # --- メッシュグリッド構築 ---
        mg_nodes, mg_inits, gx, gy = build_meshgrid_nodes("mg")
        nodes += mg_nodes
        inits += mg_inits

        # --- アフィン変換 (逆マッピング) ---
        # x_src = a*x + b*y + tx
        nodes.append(helper.make_node("Mul", ["a", gx], ["ax"]))
        nodes.append(helper.make_node("Mul", ["b", gy], ["by"]))
        nodes.append(helper.make_node("Add", ["ax", "by"], ["ax_by"]))
        nodes.append(helper.make_node("Add", ["ax_by", "tx"], ["x_src"]))

        # y_src = c*x + d*y + ty
        nodes.append(helper.make_node("Mul", ["c", gx], ["cx"]))
        nodes.append(helper.make_node("Mul", ["d", gy], ["dy"]))
        nodes.append(helper.make_node("Add", ["cx", "dy"], ["cx_dy"]))
        nodes.append(helper.make_node("Add", ["cx_dy", "ty"], ["y_src"]))

        # --- GridSample + Clip ---
        gs_nodes, gs_inits = build_gridsample_nodes("x_src", "y_src")
        nodes += gs_nodes
        inits += gs_inits

        # --- 入出力定義 ---
        inputs = [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("tx", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("d", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("ty", TensorProto.FLOAT, [1]),
        ]
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )

        return helper.make_graph(
            nodes, self.op_name, inputs, [output_vi],
            initializer=inits,
        )
