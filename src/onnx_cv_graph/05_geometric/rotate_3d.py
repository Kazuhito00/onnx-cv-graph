"""3D 回転の ONNX グラフ定義.

GridSample ベースで 3D 回転の透視投影を実現する。

画像平面を (0, 0, f) に配置し、その中心を回転軸として回転→再射影するモデル。
焦点距離 f=2.0 固定。Zoom は出力座標のスケーリング (zoom>1 で拡大)。
回転行列 R = Rz @ Ry @ Rx を構築し、画像中心基準の逆マッピングで
各出力ピクセルのソース座標を算出して GridSample(bilinear) → Clip(0,1) で出力する。

中心基準の逆マッピング (最適化版):
    ホモグラフィ係数を事前にスカラーで計算し、ピクセルレベル演算を最小化:
    fA = f*(rt22*rt00 - rt02*rt20),  fB = f*(rt22*rt01 - rt02*rt21)
    fC = f*(rt22*rt10 - rt12*rt20),  fD = f*(rt22*rt11 - rt12*rt21)
    zs  = rt20*xd/zoom + rt21*yd/zoom + rt22*f
    x_src = (fA*xd/zoom + fB*yd/zoom) / zs
    y_src = (fC*xd/zoom + fD*yd/zoom) / zs

Rotate3dOp:         パディング色なし (黒固定)
Rotate3dPadColorOp: パディング色 RGB 指定可能 (4ch GridSample で高速化)
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec

from ._grid_utils import build_gridsample_nodes, build_meshgrid_nodes

_FOCAL_LENGTH = 2.0


def _build_rotate_3d_nodes(gx, gy):
    """3D 回転の共通ノード群を構築する.

    ホモグラフィ係数をスカラーで事前計算し、ピクセルレベル演算を最小化する。

    Returns (nodes, inits, x_src_name, y_src_name)
    """
    nodes = []
    inits = []

    # --- 定数 ---
    deg2rad = numpy_helper.from_array(np.float32(np.pi / 180.0), "deg2rad")
    eps = numpy_helper.from_array(np.float32(1e-7), "eps")
    focal = numpy_helper.from_array(np.float32(_FOCAL_LENGTH), "focal")
    inits += [deg2rad, eps, focal]

    # --- 出力座標を zoom でスケーリング ---
    nodes.append(helper.make_node("Div", [gx, "zoom"], ["xd_s"]))
    nodes.append(helper.make_node("Div", [gy, "zoom"], ["yd_s"]))

    # --- 各軸の角度をラジアンに変換 ---
    for axis in ("x", "y", "z"):
        name = f"angle_{axis}"
        nodes.append(helper.make_node("Mul", [name, "deg2rad"], [f"rad_{axis}"]))
        nodes.append(helper.make_node("Cos", [f"rad_{axis}"], [f"c{axis}"]))
        nodes.append(helper.make_node("Sin", [f"rad_{axis}"], [f"s{axis}"]))

    # --- 回転行列 R = Rz @ Ry @ Rx の要素 (全スカラー演算) ---
    nodes.append(helper.make_node("Mul", ["cz", "cy"], ["cz_cy"]))
    nodes.append(helper.make_node("Mul", ["sz", "cy"], ["sz_cy"]))
    nodes.append(helper.make_node("Mul", ["cz", "sy"], ["cz_sy"]))
    nodes.append(helper.make_node("Mul", ["sz", "sy"], ["sz_sy"]))
    nodes.append(helper.make_node("Mul", ["cy", "sx"], ["cy_sx"]))
    nodes.append(helper.make_node("Mul", ["cy", "cx"], ["cy_cx"]))
    nodes.append(helper.make_node("Mul", ["cz_sy", "sx"], ["cz_sy_sx"]))
    nodes.append(helper.make_node("Mul", ["cz_sy", "cx"], ["cz_sy_cx"]))
    nodes.append(helper.make_node("Mul", ["sz_sy", "sx"], ["sz_sy_sx"]))
    nodes.append(helper.make_node("Mul", ["sz_sy", "cx"], ["sz_sy_cx"]))
    nodes.append(helper.make_node("Mul", ["sz", "cx"], ["sz_cx"]))
    nodes.append(helper.make_node("Mul", ["sz", "sx"], ["sz_sx"]))
    nodes.append(helper.make_node("Mul", ["cz", "cx"], ["cz_cx"]))
    nodes.append(helper.make_node("Mul", ["cz", "sx"], ["cz_sx"]))

    # R^T 要素
    nodes.append(helper.make_node("Neg", ["sy"], ["neg_sy"]))
    nodes.append(helper.make_node("Sub", ["cz_sy_sx", "sz_cx"], ["rt10"]))
    nodes.append(helper.make_node("Add", ["sz_sy_sx", "cz_cx"], ["rt11"]))
    nodes.append(helper.make_node("Add", ["cz_sy_cx", "sz_sx"], ["rt20"]))
    nodes.append(helper.make_node("Sub", ["sz_sy_cx", "cz_sx"], ["rt21"]))

    # --- ホモグラフィ係数の事前計算 (全スカラー演算) ---
    # A = rt22*rt00 - rt02*rt20,  B = rt22*rt01 - rt02*rt21
    # C = rt22*rt10 - rt12*rt20,  D = rt22*rt11 - rt12*rt21
    for coeff, r22_rij, r0k_rkj in [
        ("A", ("cy_cx", "cz_cy"), ("neg_sy", "rt20")),
        ("B", ("cy_cx", "sz_cy"), ("neg_sy", "rt21")),
        ("C", ("cy_cx", "rt10"),  ("cy_sx",  "rt20")),
        ("D", ("cy_cx", "rt11"),  ("cy_sx",  "rt21")),
    ]:
        nodes.append(helper.make_node("Mul", list(r22_rij), [f"{coeff}_p"]))
        nodes.append(helper.make_node("Mul", list(r0k_rkj), [f"{coeff}_q"]))
        nodes.append(helper.make_node("Sub", [f"{coeff}_p", f"{coeff}_q"], [coeff]))
        nodes.append(helper.make_node("Mul", ["focal", coeff], [f"f{coeff}"]))

    # --- zs (ピクセルレベル) ---
    nodes.append(helper.make_node("Mul", ["rt20", "xd_s"], ["rt20_x"]))
    nodes.append(helper.make_node("Mul", ["rt21", "yd_s"], ["rt21_y"]))
    nodes.append(helper.make_node("Mul", ["cy_cx", "focal"], ["rt22_f"]))
    nodes.append(helper.make_node("Add", ["rt20_x", "rt21_y"], ["zs_xy"]))
    nodes.append(helper.make_node("Add", ["zs_xy", "rt22_f"], ["zs"]))

    # ゼロ除算防止
    nodes.append(helper.make_node("Sign", ["zs"], ["zs_sign"]))
    nodes.append(helper.make_node("Abs", ["zs"], ["zs_abs"]))
    nodes.append(helper.make_node("Max", ["zs_abs", "eps"], ["zs_clamped"]))
    nodes.append(helper.make_node("Mul", ["zs_sign", "zs_clamped"], ["zs_safe"]))

    # --- x_src = (fA*xd_s + fB*yd_s) / zs_safe ---
    nodes.append(helper.make_node("Mul", ["fA", "xd_s"], ["fA_x"]))
    nodes.append(helper.make_node("Mul", ["fB", "yd_s"], ["fB_y"]))
    nodes.append(helper.make_node("Add", ["fA_x", "fB_y"], ["x_num"]))
    nodes.append(helper.make_node("Div", ["x_num", "zs_safe"], ["x_src"]))

    # --- y_src = (fC*xd_s + fD*yd_s) / zs_safe ---
    nodes.append(helper.make_node("Mul", ["fC", "xd_s"], ["fC_x"]))
    nodes.append(helper.make_node("Mul", ["fD", "yd_s"], ["fD_y"]))
    nodes.append(helper.make_node("Add", ["fC_x", "fD_y"], ["y_num"]))
    nodes.append(helper.make_node("Div", ["y_num", "zs_safe"], ["y_src"]))

    return nodes, inits, "x_src", "y_src"


class Rotate3dOp(OnnxGraphOp):
    """3D 回転 (GridSample ベース, 黒パディング).

    X/Y/Z 軸回転と Zoom 率で画像中心基準の 3D 回転透視投影を適用する。
    はみ出し部分は黒 (0,0,0) でパディングされる。
    """

    @property
    def op_name(self) -> str:
        return "rotate_3d"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("angle_x", TensorProto.FLOAT, [1]),
            ("angle_y", TensorProto.FLOAT, [1]),
            ("angle_z", TensorProto.FLOAT, [1]),
            ("zoom", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {
            "angle_x": (-180.0, 180.0, 0.0),
            "angle_y": (-180.0, 180.0, 0.0),
            "angle_z": (-180.0, 180.0, 0.0),
            "zoom": (0.1, 5.0, 1.0),
        }

    def build_graph(self) -> GraphProto:
        nodes = []
        inits = []

        mg_nodes, mg_inits, gx, gy = build_meshgrid_nodes("mg")
        nodes += mg_nodes
        inits += mg_inits

        rot_nodes, rot_inits, _, _ = _build_rotate_3d_nodes(gx, gy)
        nodes += rot_nodes
        inits += rot_inits

        gs_nodes, gs_inits = build_gridsample_nodes("x_src", "y_src")
        nodes += gs_nodes
        inits += gs_inits

        inputs = [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["N", 3, "H", "W"]
            ),
            helper.make_tensor_value_info("angle_x", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("angle_y", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("angle_z", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("zoom", TensorProto.FLOAT, [1]),
        ]
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )

        return helper.make_graph(
            nodes, self.op_name, inputs, [output_vi],
            initializer=inits,
        )


class Rotate3dPadColorOp(OnnxGraphOp):
    """3D 回転 (GridSample ベース, パディング色 RGB 指定).

    X/Y/Z 軸回転と Zoom 率で画像中心基準の 3D 回転透視投影を適用する。
    はみ出し部分は指定 RGB 色でパディングされる。
    入力 3ch + alpha 1ch = 4ch を単一の GridSample で処理して高速化。
    """

    @property
    def op_name(self) -> str:
        return "rotate_3d_pad_color"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("angle_x", TensorProto.FLOAT, [1]),
            ("angle_y", TensorProto.FLOAT, [1]),
            ("angle_z", TensorProto.FLOAT, [1]),
            ("zoom", TensorProto.FLOAT, [1]),
            ("pad_r", TensorProto.FLOAT, [1]),
            ("pad_g", TensorProto.FLOAT, [1]),
            ("pad_b", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {
            "angle_x": (-180.0, 180.0, 0.0),
            "angle_y": (-180.0, 180.0, 0.0),
            "angle_z": (-180.0, 180.0, 0.0),
            "zoom": (0.1, 5.0, 1.0),
            "pad_r": (0.0, 1.0, 0.0),
            "pad_g": (0.0, 1.0, 0.0),
            "pad_b": (0.0, 1.0, 0.0),
        }

    def build_graph(self) -> GraphProto:
        nodes = []
        inits = []

        mg_nodes, mg_inits, gx, gy = build_meshgrid_nodes("mg")
        nodes += mg_nodes
        inits += mg_inits

        rot_nodes, rot_inits, _, _ = _build_rotate_3d_nodes(gx, gy)
        nodes += rot_nodes
        inits += rot_inits

        one_f = numpy_helper.from_array(np.float32(1.0), "one_f")
        inits.append(one_f)

        # --- 4ch 入力の構築: [N, 3, H, W] + [N, 1, H, W] → [N, 4, H, W] ---
        ones_scalar = numpy_helper.from_array(
            np.array([[[[1.0]]]], dtype=np.float32), "ones_scalar",
        )
        inits.append(ones_scalar)
        nodes.append(helper.make_node("Shape", ["input"], ["pc_in_shape"]))

        idx_n = numpy_helper.from_array(np.array([0], dtype=np.int64), "idx_n")
        idx_hw = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "idx_hw")
        const_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), "const_1")
        inits += [idx_n, idx_hw, const_1]

        nodes.append(helper.make_node(
            "Gather", ["pc_in_shape", "idx_n"], ["n_dim"], axis=0,
        ))
        nodes.append(helper.make_node("Reshape", ["n_dim", "const_1"], ["n_1d"]))
        nodes.append(helper.make_node(
            "Gather", ["pc_in_shape", "idx_hw"], ["hw_dims"], axis=0,
        ))
        nodes.append(helper.make_node(
            "Concat", ["n_1d", "const_1", "hw_dims"], ["alpha_in_shape"], axis=0,
        ))
        nodes.append(helper.make_node(
            "Expand", ["ones_scalar", "alpha_in_shape"], ["ones_ch"],
        ))
        # Concat: [N,3,H,W] + [N,1,H,W] → [N,4,H,W]
        nodes.append(helper.make_node(
            "Concat", ["input", "ones_ch"], ["input_4ch"], axis=1,
        ))

        # --- Grid 構築: x_src, y_src → [N, H, W, 2] ---
        axes_neg1 = numpy_helper.from_array(
            np.array([-1], dtype=np.int64), "gs_ax_neg1"
        )
        axes_0 = numpy_helper.from_array(
            np.array([0], dtype=np.int64), "gs_ax_0"
        )
        inits += [axes_neg1, axes_0]

        nodes.append(helper.make_node(
            "Unsqueeze", ["x_src", "gs_ax_neg1"], ["gs_x_3d"],
        ))
        nodes.append(helper.make_node(
            "Unsqueeze", ["y_src", "gs_ax_neg1"], ["gs_y_3d"],
        ))
        nodes.append(helper.make_node(
            "Concat", ["gs_x_3d", "gs_y_3d"], ["gs_grid_hw2"], axis=-1,
        ))
        nodes.append(helper.make_node(
            "Unsqueeze", ["gs_grid_hw2", "gs_ax_0"], ["gs_grid_1hw2"],
        ))

        idx_nhw = numpy_helper.from_array(
            np.array([0, 2, 3], dtype=np.int64), "gs_idx_nhw"
        )
        const_2 = numpy_helper.from_array(
            np.array([2], dtype=np.int64), "gs_c2"
        )
        inits += [idx_nhw, const_2]
        nodes.append(helper.make_node(
            "Gather", ["pc_in_shape", "gs_idx_nhw"], ["gs_nhw"], axis=0,
        ))
        nodes.append(helper.make_node(
            "Concat", ["gs_nhw", "gs_c2"], ["gs_target_shape"], axis=0,
        ))
        nodes.append(helper.make_node(
            "Expand", ["gs_grid_1hw2", "gs_target_shape"], ["gs_grid"],
        ))

        # --- 単一 GridSample: [N, 4, H, W] → [N, 4, H, W] ---
        nodes.append(helper.make_node(
            "GridSample", ["input_4ch", "gs_grid"], ["gs_sampled_4ch"],
            mode="bilinear", padding_mode="zeros", align_corners=1,
        ))

        # --- Split → sampled [N, 3, H, W] + alpha [N, 1, H, W] ---
        split_lens = numpy_helper.from_array(
            np.array([3, 1], dtype=np.int64), "split_lens",
        )
        inits.append(split_lens)
        nodes.append(helper.make_node(
            "Split", ["gs_sampled_4ch", "split_lens"],
            ["gs_sampled", "alpha"], axis=1,
        ))

        # --- パディング色テンソル [1, 3, 1, 1] ---
        nodes.append(helper.make_node(
            "Concat", ["pad_r", "pad_g", "pad_b"], ["pad_rgb_flat"], axis=0,
        ))
        sh_1_3_1_1 = numpy_helper.from_array(
            np.array([1, 3, 1, 1], dtype=np.int64), "sh_1_3_1_1",
        )
        inits.append(sh_1_3_1_1)
        nodes.append(helper.make_node(
            "Reshape", ["pad_rgb_flat", "sh_1_3_1_1"], ["pad_color"],
        ))

        # --- 合成: output = sampled + pad_color * (1 - alpha) ---
        nodes.append(helper.make_node("Sub", ["one_f", "alpha"], ["inv_alpha"]))
        nodes.append(helper.make_node("Mul", ["pad_color", "inv_alpha"], ["bg"]))
        nodes.append(helper.make_node("Add", ["gs_sampled", "bg"], ["blended"]))

        clip_min = numpy_helper.from_array(np.float32(0.0), "clip_min")
        clip_max = numpy_helper.from_array(np.float32(1.0), "clip_max")
        inits += [clip_min, clip_max]
        nodes.append(helper.make_node(
            "Clip", ["blended", "clip_min", "clip_max"], ["output"],
        ))

        inputs = [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["N", 3, "H", "W"]
            ),
            helper.make_tensor_value_info("angle_x", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("angle_y", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("angle_z", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("zoom", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("pad_r", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("pad_g", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("pad_b", TensorProto.FLOAT, [1]),
        ]
        output_vi = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["N", 3, "H", "W"]
        )

        return helper.make_graph(
            nodes, self.op_name, inputs, [output_vi],
            initializer=inits,
        )
