"""カラーマップ適用の ONNX グラフ定義.

グレースケール化 → 256段階に量子化 → LUT (Gather) で RGB カラーマップを適用する.
OpenCV の applyColorMap 相当. カラーマップごとに別モデルを生成する.
"""

from typing import List

import cv2
import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec

# (名前, OpenCV 定数) のペア
COLORMAPS = [
    ("jet", cv2.COLORMAP_JET),
    ("turbo", cv2.COLORMAP_TURBO),
    ("inferno", cv2.COLORMAP_INFERNO),
    ("viridis", cv2.COLORMAP_VIRIDIS),
]


def _build_lut(colormap_id: int) -> np.ndarray:
    """OpenCV のカラーマップ定数から (256, 3) の RGB float32 LUT を生成する."""
    gray_bar = np.arange(256, dtype=np.uint8).reshape(256, 1)
    bgr = cv2.applyColorMap(gray_bar, colormap_id)  # (256, 1, 3) BGR uint8
    rgb = bgr[:, 0, ::-1].copy()  # (256, 3) RGB uint8
    return rgb.astype(np.float32) / 255.0


class ColormapOp(OnnxGraphOp):
    """カラーマップ適用.

    グレースケール化 → 255倍 → Floor → Cast(int64) → Clip(0,255) →
    Squeeze → Gather(LUT) → Transpose(NHWC→NCHW).
    """

    def __init__(self, name: str = "jet", colormap_id: int = cv2.COLORMAP_JET):
        self._cmap_name = name
        self._colormap_id = colormap_id

    @property
    def op_name(self) -> str:
        return f"colormap_{self._cmap_name}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        return [cls(name, cid) for name, cid in COLORMAPS]

    def build_graph(self) -> GraphProto:
        # 定数
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        weights_init = numpy_helper.from_array(weights, name="luma_weights")
        axes = np.array([1], dtype=np.int64)
        axes_init = numpy_helper.from_array(axes, name="axes")
        scale = np.array([255.0], dtype=np.float32)
        scale_init = numpy_helper.from_array(scale, name="scale_255")
        zero_i = np.array([0], dtype=np.int64)
        zero_i_init = numpy_helper.from_array(zero_i, name="zero_i")
        max_i = np.array([255], dtype=np.int64)
        max_i_init = numpy_helper.from_array(max_i, name="max_i")

        # squeeze 用 axes
        sq_axes = np.array([1], dtype=np.int64)
        sq_axes_init = numpy_helper.from_array(sq_axes, name="sq_axes")

        # LUT テーブル (256, 3) float32
        lut = _build_lut(self._colormap_id)
        lut_init = numpy_helper.from_array(lut, name="lut")

        # transpose 用 perm: (N, H, W, 3) → (N, 3, H, W)
        # (attribute なので initializer 不要)

        initializers = [weights_init, axes_init, scale_init,
                        zero_i_init, max_i_init, sq_axes_init, lut_init]

        nodes = []

        # 1. グレースケール化: (N, 3, H, W) → (N, 1, H, W)
        nodes.append(helper.make_node("Mul", ["input", "luma_weights"], ["weighted"]))
        nodes.append(helper.make_node("ReduceSum", ["weighted", "axes"], ["gray"], keepdims=1))

        # 2. 量子化: gray * 255 → Floor → Cast(int64) → Clip(0, 255)
        nodes.append(helper.make_node("Mul", ["gray", "scale_255"], ["gray_scaled"]))
        nodes.append(helper.make_node("Floor", ["gray_scaled"], ["gray_floor"]))
        nodes.append(helper.make_node("Cast", ["gray_floor"], ["gray_i64"], to=TensorProto.INT64))
        nodes.append(helper.make_node("Clip", ["gray_i64", "zero_i", "max_i"], ["indices_4d"]))

        # 3. Squeeze: (N, 1, H, W) → (N, H, W) — Gather が4D indices + 2D data で5D出力にならないように
        nodes.append(helper.make_node("Squeeze", ["indices_4d", "sq_axes"], ["indices"]))

        # 4. Gather: LUT(256, 3) × indices(N, H, W) → (N, H, W, 3)
        nodes.append(helper.make_node("Gather", ["lut", "indices"], ["nhwc"], axis=0))

        # 5. Transpose: (N, H, W, 3) → (N, 3, H, W)
        nodes.append(helper.make_node("Transpose", ["nhwc"], ["output"], perm=[0, 3, 1, 2]))

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes,
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=initializers,
        )
