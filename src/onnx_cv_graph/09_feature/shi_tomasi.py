"""Shi-Tomasi コーナースコアの ONNX グラフ定義.

OpenCV の cornerMinEigenVal 相当. Sobel 微分 → 構造テンソル → 窓関数 →
最小固有値 λ_min = (trace/2) - sqrt((trace/2)² - det) を計算し、
[0, 1] に正規化して出力する.
block_size ごとに別モデルを生成する (3/5).
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class ShiTomasiOp(OnnxGraphOp):
    """Shi-Tomasi コーナースコア (最小固有値).

    グレースケール化 → Sobel 微分 (Ix, Iy) → 構造テンソル (Ix², Iy², Ix*Iy) →
    box filter (AveragePool) → λ_min = (trace/2) - sqrt((trace/2)² - det) →
    MinMax 正規化 → Clip(0,1) → 3ch 拡張.
    """

    def __init__(self, block_size: int = 3):
        self._block_size = block_size

    @property
    def op_name(self) -> str:
        return f"shi_tomasi_{self._block_size}x{self._block_size}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

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
        sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32).reshape(1, 1, 3, 3)
        sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32).reshape(1, 1, 3, 3)
        sx_init = numpy_helper.from_array(sx, name="sobel_x")
        sy_init = numpy_helper.from_array(sy, name="sobel_y")

        sobel_pads = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
        sobel_pads_init = numpy_helper.from_array(sobel_pads, name="sobel_pads")

        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_shape_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        zero = np.array([0.0], dtype=np.float32)
        one = np.array([1.0], dtype=np.float32)
        eps = np.array([1e-7], dtype=np.float32)
        half = np.array([0.5], dtype=np.float32)
        zero_init = numpy_helper.from_array(zero, name="zero")
        one_init = numpy_helper.from_array(one, name="one")
        eps_init = numpy_helper.from_array(eps, name="eps")
        half_init = numpy_helper.from_array(half, name="half")

        initializers = [
            weights_init, axes_init, sx_init, sy_init, sobel_pads_init,
            expand_shape_init, zero_init, one_init, eps_init, half_init,
        ]

        nodes = []

        # 1. グレースケール化
        nodes.append(helper.make_node("Mul", ["input", "luma_weights"], ["weighted"]))
        nodes.append(helper.make_node("ReduceSum", ["weighted", "axes"], ["gray"], keepdims=1))

        # 2. Sobel 微分
        nodes.append(helper.make_node("Pad", ["gray", "sobel_pads"], ["gray_padded"], mode="reflect"))
        nodes.append(helper.make_node("Conv", ["gray_padded", "sobel_x"], ["Ix"]))
        nodes.append(helper.make_node("Conv", ["gray_padded", "sobel_y"], ["Iy"]))

        # 3. 構造テンソル
        nodes.append(helper.make_node("Mul", ["Ix", "Ix"], ["Ix2"]))
        nodes.append(helper.make_node("Mul", ["Iy", "Iy"], ["Iy2"]))
        nodes.append(helper.make_node("Mul", ["Ix", "Iy"], ["IxIy"]))

        # 4. 窓関数 (box filter)
        pool_attrs = dict(kernel_shape=[bs, bs], pads=[bs_pad, bs_pad, bs_pad, bs_pad])
        nodes.append(helper.make_node("AveragePool", ["Ix2"], ["Sxx"], **pool_attrs))
        nodes.append(helper.make_node("AveragePool", ["Iy2"], ["Syy"], **pool_attrs))
        nodes.append(helper.make_node("AveragePool", ["IxIy"], ["Sxy"], **pool_attrs))

        # 5. 最小固有値: λ_min = (trace/2) - sqrt(max(0, (trace/2)² - det))
        #    det = Sxx*Syy - Sxy²
        #    trace = Sxx + Syy
        nodes.append(helper.make_node("Mul", ["Sxx", "Syy"], ["SxxSyy"]))
        nodes.append(helper.make_node("Mul", ["Sxy", "Sxy"], ["Sxy2"]))
        nodes.append(helper.make_node("Sub", ["SxxSyy", "Sxy2"], ["det"]))
        nodes.append(helper.make_node("Add", ["Sxx", "Syy"], ["trace"]))
        nodes.append(helper.make_node("Mul", ["trace", "half"], ["half_trace"]))
        nodes.append(helper.make_node("Mul", ["half_trace", "half_trace"], ["half_trace2"]))
        nodes.append(helper.make_node("Sub", ["half_trace2", "det"], ["discriminant"]))
        # sqrt 内が負にならないよう Relu で下限 0
        nodes.append(helper.make_node("Relu", ["discriminant"], ["disc_safe"]))
        nodes.append(helper.make_node("Sqrt", ["disc_safe"], ["sqrt_disc"]))
        nodes.append(helper.make_node("Sub", ["half_trace", "sqrt_disc"], ["lambda_min"]))

        # 6. 正の応答のみ残す
        nodes.append(helper.make_node("Relu", ["lambda_min"], ["lm_pos"]))

        # 7. MinMax 正規化
        nodes.append(helper.make_node("ReduceMin", ["lm_pos"], ["lm_min"], axes=[0, 1, 2, 3], keepdims=1))
        nodes.append(helper.make_node("ReduceMax", ["lm_pos"], ["lm_max"], axes=[0, 1, 2, 3], keepdims=1))
        nodes.append(helper.make_node("Sub", ["lm_pos", "lm_min"], ["lm_shifted"]))
        nodes.append(helper.make_node("Sub", ["lm_max", "lm_min"], ["lm_range"]))
        nodes.append(helper.make_node("Add", ["lm_range", "eps"], ["lm_range_safe"]))
        nodes.append(helper.make_node("Div", ["lm_shifted", "lm_range_safe"], ["lm_norm"]))

        # 8. Clip(0, 1) → 3ch 拡張
        nodes.append(helper.make_node("Clip", ["lm_norm", "zero", "one"], ["clipped"]))
        nodes.append(helper.make_node("Expand", ["clipped", "expand_shape"], ["output"]))

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes,
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=initializers,
        )
