"""Harris コーナー検出の ONNX グラフ定義.

OpenCV の cornerHarris 相当. Sobel 微分 → 構造テンソル → 窓関数 (box filter) →
Harris 応答 R = det(M) - k * trace(M)² を計算し、[0, 1] に正規化して出力する.
block_size ごとに別モデルを生成する (3/5). k は推論時パラメータ.
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


def _sobel_kernel_x() -> np.ndarray:
    """Sobel X カーネル (3x3)."""
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32).reshape(1, 1, 3, 3)


def _sobel_kernel_y() -> np.ndarray:
    """Sobel Y カーネル (3x3)."""
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32).reshape(1, 1, 3, 3)


class HarrisCornerOp(OnnxGraphOp):
    """Harris コーナー検出.

    グレースケール化 → Sobel 微分 (Ix, Iy) → 構造テンソル (Ix², Iy², Ix*Iy) →
    box filter (AveragePool) → R = det - k * trace² → MinMax 正規化 → Clip(0,1) → 3ch 拡張.
    """

    def __init__(self, block_size: int = 3):
        self._block_size = block_size

    @property
    def op_name(self) -> str:
        return f"harris_corner_{self._block_size}x{self._block_size}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("k", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"k": (0.01, 0.2, 0.04)}

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        return [cls(3), cls(5)]

    def build_graph(self) -> GraphProto:
        bs = self._block_size
        bs_pad = bs // 2

        # 定数
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        weights_init = numpy_helper.from_array(weights, name="luma_weights")
        axes = np.array([1], dtype=np.int64)
        axes_init = numpy_helper.from_array(axes, name="axes")

        # Sobel カーネル
        sx = _sobel_kernel_x()
        sy = _sobel_kernel_y()
        sx_init = numpy_helper.from_array(sx, name="sobel_x")
        sy_init = numpy_helper.from_array(sy, name="sobel_y")

        # Sobel 用 reflect パディング
        sobel_pads = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
        sobel_pads_init = numpy_helper.from_array(sobel_pads, name="sobel_pads")

        # 3ch 拡張用
        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_shape_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        # Clip 用定数
        zero = np.array([0.0], dtype=np.float32)
        one = np.array([1.0], dtype=np.float32)
        zero_init = numpy_helper.from_array(zero, name="zero")
        one_init = numpy_helper.from_array(one, name="one")

        # epsilon (ゼロ除算回避)
        eps = np.array([1e-7], dtype=np.float32)
        eps_init = numpy_helper.from_array(eps, name="eps")

        initializers = [
            weights_init, axes_init, sx_init, sy_init, sobel_pads_init,
            expand_shape_init, zero_init, one_init, eps_init,
        ]

        nodes = []

        # 1. グレースケール化: input(N,3,H,W) → gray(N,1,H,W)
        nodes.append(helper.make_node("Mul", ["input", "luma_weights"], ["weighted"]))
        nodes.append(helper.make_node("ReduceSum", ["weighted", "axes"], ["gray"], keepdims=1))

        # 2. Sobel 微分: Pad(reflect) → Conv
        nodes.append(helper.make_node("Pad", ["gray", "sobel_pads"], ["gray_padded"], mode="reflect"))
        nodes.append(helper.make_node("Conv", ["gray_padded", "sobel_x"], ["Ix"]))
        nodes.append(helper.make_node("Conv", ["gray_padded", "sobel_y"], ["Iy"]))

        # 3. 構造テンソル要素: Ix², Iy², Ix*Iy
        nodes.append(helper.make_node("Mul", ["Ix", "Ix"], ["Ix2"]))
        nodes.append(helper.make_node("Mul", ["Iy", "Iy"], ["Iy2"]))
        nodes.append(helper.make_node("Mul", ["Ix", "Iy"], ["IxIy"]))

        # 4. 窓関数 (box filter): AveragePool で block_size の領域平均
        pool_attrs = dict(kernel_shape=[bs, bs], pads=[bs_pad, bs_pad, bs_pad, bs_pad])
        nodes.append(helper.make_node("AveragePool", ["Ix2"], ["Sxx"], **pool_attrs))
        nodes.append(helper.make_node("AveragePool", ["Iy2"], ["Syy"], **pool_attrs))
        nodes.append(helper.make_node("AveragePool", ["IxIy"], ["Sxy"], **pool_attrs))

        # 5. Harris 応答: R = det(M) - k * trace(M)²
        #    det(M) = Sxx * Syy - Sxy * Sxy
        #    trace(M) = Sxx + Syy
        nodes.append(helper.make_node("Mul", ["Sxx", "Syy"], ["SxxSyy"]))
        nodes.append(helper.make_node("Mul", ["Sxy", "Sxy"], ["Sxy2"]))
        nodes.append(helper.make_node("Sub", ["SxxSyy", "Sxy2"], ["det"]))
        nodes.append(helper.make_node("Add", ["Sxx", "Syy"], ["trace"]))
        nodes.append(helper.make_node("Mul", ["trace", "trace"], ["trace2"]))
        nodes.append(helper.make_node("Mul", ["k", "trace2"], ["k_trace2"]))
        nodes.append(helper.make_node("Sub", ["det", "k_trace2"], ["R"]))

        # 6. 正の応答のみ残す (コーナー応答は正値)
        nodes.append(helper.make_node("Relu", ["R"], ["R_pos"]))

        # 7. MinMax 正規化: (R_pos - min) / (max - min + eps) → [0, 1]
        nodes.append(helper.make_node("ReduceMin", ["R_pos"], ["R_min"], axes=[0, 1, 2, 3], keepdims=1))
        nodes.append(helper.make_node("ReduceMax", ["R_pos"], ["R_max"], axes=[0, 1, 2, 3], keepdims=1))
        nodes.append(helper.make_node("Sub", ["R_pos", "R_min"], ["R_shifted"]))
        nodes.append(helper.make_node("Sub", ["R_max", "R_min"], ["R_range"]))
        nodes.append(helper.make_node("Add", ["R_range", "eps"], ["R_range_safe"]))
        nodes.append(helper.make_node("Div", ["R_shifted", "R_range_safe"], ["R_norm"]))

        # 8. Clip(0, 1) → 3ch 拡張
        nodes.append(helper.make_node("Clip", ["R_norm", "zero", "one"], ["clipped"]))
        nodes.append(helper.make_node("Expand", ["clipped", "expand_shape"], ["output"]))

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        k_vi = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes,
            self.op_name,
            [input_vi, k_vi],
            [output_vi],
            initializer=initializers,
        )
