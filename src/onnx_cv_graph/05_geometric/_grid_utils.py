"""GridSample ベースの幾何変換で共有するメッシュグリッド構築ヘルパー.

入力画像の Shape から [-1, 1] 正規化座標のメッシュグリッドを構築し、
変換後の座標を GridSample に渡して出力を得る共通処理を提供する。
"""

from typing import List, Tuple

import numpy as np
from onnx import NodeProto, TensorProto
from onnx import helper as H
from onnx import numpy_helper as nph


def build_meshgrid_nodes(
    prefix: str = "mg",
) -> Tuple[List[NodeProto], list, str, str]:
    """入力画像の H, W から [-1, 1] 正規化メッシュグリッドを構築する.

    入力テンソル名は ``"input"`` 固定。Shape → Gather(H,W) → Range → 正規化 →
    Reshape + Expand により (H, W) の grid_x, grid_y を生成する。

    Parameters
    ----------
    prefix : str
        ノード・テンソル名の衝突回避用プレフィックス。

    Returns
    -------
    nodes : list[NodeProto]
    initializers : list[TensorProto]
    grid_x_name : str
        W 方向の正規化座標テンソル名 (shape [H, W])。
    grid_y_name : str
        H 方向の正規化座標テンソル名 (shape [H, W])。
    """
    p = f"{prefix}_"
    nodes: List[NodeProto] = []
    inits: list = []

    # --- Shape(input) → [N, C, H, W] ---
    nodes.append(H.make_node("Shape", ["input"], [f"{p}shape"]))

    idx2 = nph.from_array(np.array(2, dtype=np.int64), f"{p}idx2")
    idx3 = nph.from_array(np.array(3, dtype=np.int64), f"{p}idx3")
    inits += [idx2, idx3]

    nodes.append(H.make_node("Gather", [f"{p}shape", f"{p}idx2"], [f"{p}H_i64"], axis=0))
    nodes.append(H.make_node("Gather", [f"{p}shape", f"{p}idx3"], [f"{p}W_i64"], axis=0))

    # int64 → float32
    nodes.append(H.make_node("Cast", [f"{p}H_i64"], [f"{p}H_f"], to=TensorProto.FLOAT))
    nodes.append(H.make_node("Cast", [f"{p}W_i64"], [f"{p}W_f"], to=TensorProto.FLOAT))

    # --- 定数 ---
    zero_f = nph.from_array(np.float32(0.0), f"{p}zero_f")
    one_f = nph.from_array(np.float32(1.0), f"{p}one_f")
    two_f = nph.from_array(np.float32(2.0), f"{p}two_f")
    inits += [zero_f, one_f, two_f]

    # --- Range ---
    nodes.append(H.make_node("Range", [f"{p}zero_f", f"{p}H_f", f"{p}one_f"], [f"{p}y_range"]))
    nodes.append(H.make_node("Range", [f"{p}zero_f", f"{p}W_f", f"{p}one_f"], [f"{p}x_range"]))

    # --- 正規化: val * 2/(dim-1) - 1 ---
    # y 方向 (H)
    nodes.append(H.make_node("Sub", [f"{p}H_f", f"{p}one_f"], [f"{p}H_m1"]))
    nodes.append(H.make_node("Div", [f"{p}two_f", f"{p}H_m1"], [f"{p}y_scale"]))
    nodes.append(H.make_node("Mul", [f"{p}y_range", f"{p}y_scale"], [f"{p}y_scaled"]))
    nodes.append(H.make_node("Sub", [f"{p}y_scaled", f"{p}one_f"], [f"{p}y_norm"]))

    # x 方向 (W)
    nodes.append(H.make_node("Sub", [f"{p}W_f", f"{p}one_f"], [f"{p}W_m1"]))
    nodes.append(H.make_node("Div", [f"{p}two_f", f"{p}W_m1"], [f"{p}x_scale"]))
    nodes.append(H.make_node("Mul", [f"{p}x_range", f"{p}x_scale"], [f"{p}x_scaled"]))
    nodes.append(H.make_node("Sub", [f"{p}x_scaled", f"{p}one_f"], [f"{p}x_norm"]))

    # --- Reshape + Expand → (H, W) ---
    shape_neg1_1 = nph.from_array(np.array([-1, 1], dtype=np.int64), f"{p}sh_n1_1")
    shape_1_neg1 = nph.from_array(np.array([1, -1], dtype=np.int64), f"{p}sh_1_n1")
    inits += [shape_neg1_1, shape_1_neg1]

    # y_norm [H] → [H, 1]
    nodes.append(H.make_node("Reshape", [f"{p}y_norm", f"{p}sh_n1_1"], [f"{p}y_col"]))
    # x_norm [W] → [1, W]
    nodes.append(H.make_node("Reshape", [f"{p}x_norm", f"{p}sh_1_n1"], [f"{p}x_row"]))

    # Expand 用の [H, W] 形状テンソルを構築
    shape_1i = nph.from_array(np.array([1], dtype=np.int64), f"{p}sh1i")
    inits.append(shape_1i)
    nodes.append(H.make_node("Reshape", [f"{p}H_i64", f"{p}sh1i"], [f"{p}H_1d"]))
    nodes.append(H.make_node("Reshape", [f"{p}W_i64", f"{p}sh1i"], [f"{p}W_1d"]))
    nodes.append(H.make_node("Concat", [f"{p}H_1d", f"{p}W_1d"], [f"{p}hw_shape"], axis=0))

    nodes.append(H.make_node("Expand", [f"{p}y_col", f"{p}hw_shape"], [f"{p}grid_y"]))
    nodes.append(H.make_node("Expand", [f"{p}x_row", f"{p}hw_shape"], [f"{p}grid_x"]))

    return nodes, inits, f"{p}grid_x", f"{p}grid_y"


def build_gridsample_nodes(
    x_src_name: str,
    y_src_name: str,
    output_name: str = "output",
    prefix: str = "gs",
) -> Tuple[List[NodeProto], list]:
    """変換済み座標から GridSample + Clip(0,1) を実行するノード群を構築する.

    Parameters
    ----------
    x_src_name : str
        変換後の x 座標テンソル名 (shape [H, W], [-1,1] 正規化)。
    y_src_name : str
        変換後の y 座標テンソル名 (shape [H, W], [-1,1] 正規化)。
    output_name : str
        出力テンソル名。
    prefix : str
        名前衝突回避用プレフィックス。

    Returns
    -------
    nodes : list[NodeProto]
    initializers : list[TensorProto]
    """
    p = f"{prefix}_"
    nodes: List[NodeProto] = []
    inits: list = []

    # --- x_src, y_src を [H, W, 1] に拡張して [H, W, 2] に結合 ---
    axes_neg1 = nph.from_array(np.array([-1], dtype=np.int64), f"{p}ax_neg1")
    inits.append(axes_neg1)

    nodes.append(H.make_node("Unsqueeze", [x_src_name, f"{p}ax_neg1"], [f"{p}x_3d"]))
    nodes.append(H.make_node("Unsqueeze", [y_src_name, f"{p}ax_neg1"], [f"{p}y_3d"]))
    nodes.append(H.make_node("Concat", [f"{p}x_3d", f"{p}y_3d"], [f"{p}grid_hw2"], axis=-1))

    # --- バッチ次元追加: [1, H, W, 2] ---
    axes_0 = nph.from_array(np.array([0], dtype=np.int64), f"{p}ax_0")
    inits.append(axes_0)
    nodes.append(H.make_node("Unsqueeze", [f"{p}grid_hw2", f"{p}ax_0"], [f"{p}grid_1hw2"]))

    # --- Expand → [N, H, W, 2] ---
    # Shape(input) から N, H, W を取得
    nodes.append(H.make_node("Shape", ["input"], [f"{p}in_shape"]))
    idx_nhw = nph.from_array(np.array([0, 2, 3], dtype=np.int64), f"{p}idx_nhw")
    inits.append(idx_nhw)
    nodes.append(H.make_node("Gather", [f"{p}in_shape", f"{p}idx_nhw"], [f"{p}nhw"], axis=0))

    const_2 = nph.from_array(np.array([2], dtype=np.int64), f"{p}c2")
    inits.append(const_2)
    nodes.append(H.make_node("Concat", [f"{p}nhw", f"{p}c2"], [f"{p}target_shape"], axis=0))
    nodes.append(H.make_node("Expand", [f"{p}grid_1hw2", f"{p}target_shape"], [f"{p}grid"]))

    # --- GridSample (bilinear, zeros, align_corners=1) ---
    nodes.append(H.make_node(
        "GridSample", ["input", f"{p}grid"], [f"{p}sampled"],
        mode="bilinear", padding_mode="zeros", align_corners=1,
    ))

    # --- Clip(0, 1) ---
    clip_min = nph.from_array(np.float32(0.0), f"{p}clip_min")
    clip_max = nph.from_array(np.float32(1.0), f"{p}clip_max")
    inits += [clip_min, clip_max]
    nodes.append(H.make_node("Clip", [f"{p}sampled", f"{p}clip_min", f"{p}clip_max"], [output_name]))

    return nodes, inits
