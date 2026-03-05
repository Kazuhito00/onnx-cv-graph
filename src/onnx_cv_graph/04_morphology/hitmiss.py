"""ヒットオアミス変換の ONNX グラフ定義.

二値画像に対して、前景カーネル (十字型) にマッチし、かつ
背景カーネル (十字型の反転) にもマッチする画素を検出する.
output = erode(input, fg_kernel) AND erode(1 - input, bg_kernel)

簡易実装として 3×3 十字型 (cross) 構造要素を使用:
  fg_kernel (十字): [[0,1,0],[1,1,1],[0,1,0]]
  bg_kernel (角):   [[1,0,1],[0,0,0],[1,0,1]]

ONNX には MinPool がないため、収縮は Neg→MaxPool→Neg で実現するが、
ここでは矩形でない構造要素のため Conv + 閾値判定で実現する.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class HitMissOp(OnnxGraphOp):
    """ヒットオアミス変換 (3×3 十字型構造要素).

    Conv(fg) → Equal(sum_fg) → AND → Conv(bg) → Equal(sum_bg) → 3ch float.
    """

    @property
    def op_name(self) -> str:
        return "hitmiss_3x3"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        # 十字型の前景カーネル
        fg = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.float32)
        # 角の背景カーネル
        bg = np.array([[1, 0, 1],
                       [0, 0, 0],
                       [1, 0, 1]], dtype=np.float32)

        fg_sum = float(fg.sum())  # 5.0
        bg_sum = float(bg.sum())  # 4.0

        # グレースケール変換用 luma
        luma = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        luma_init = numpy_helper.from_array(luma, name="luma")
        axes = np.array([1], dtype=np.int64)
        axes_init = numpy_helper.from_array(axes, name="axes")

        # Conv カーネル (1ch → 1ch): shape (1, 1, 3, 3)
        fg_kernel = fg.reshape(1, 1, 3, 3)
        bg_kernel = bg.reshape(1, 1, 3, 3)
        fg_init = numpy_helper.from_array(fg_kernel, name="fg_kernel")
        bg_init = numpy_helper.from_array(bg_kernel, name="bg_kernel")

        fg_sum_t = numpy_helper.from_array(np.array([fg_sum], dtype=np.float32), name="fg_sum")
        bg_sum_t = numpy_helper.from_array(np.array([bg_sum], dtype=np.float32), name="bg_sum")
        one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")
        threshold = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="threshold")

        expand_shape = np.array([1, 3, 1, 1], dtype=np.int64)
        expand_init = numpy_helper.from_array(expand_shape, name="expand_shape")

        nodes = [
            # グレースケール化 (二値画像想定だが念のため)
            helper.make_node("Mul", ["input", "luma"], ["weighted"]),
            helper.make_node("ReduceSum", ["weighted", "axes"], ["gray"], keepdims=1),

            # 二値化: >= 0.5 → 1.0, < 0.5 → 0.0
            helper.make_node("GreaterOrEqual", ["gray", "threshold"], ["bin_mask"]),
            helper.make_node("Cast", ["bin_mask"], ["bin_f"], to=TensorProto.FLOAT),

            # 前景の収縮: Conv(bin, fg_kernel) == fg_sum → 全画素が1の場合のみマッチ
            helper.make_node("Conv", ["bin_f", "fg_kernel"], ["fg_conv"], pads=[1, 1, 1, 1]),
            helper.make_node("Equal", ["fg_conv", "fg_sum"], ["fg_match"]),

            # 背景: 1 - bin
            helper.make_node("Sub", ["one", "bin_f"], ["inv_bin"]),

            # 背景の収縮: Conv(inv_bin, bg_kernel) == bg_sum
            helper.make_node("Conv", ["inv_bin", "bg_kernel"], ["bg_conv"], pads=[1, 1, 1, 1]),
            helper.make_node("Equal", ["bg_conv", "bg_sum"], ["bg_match"]),

            # AND: fg_match AND bg_match
            helper.make_node("And", ["fg_match", "bg_match"], ["hit"]),
            helper.make_node("Cast", ["hit"], ["hit_f"], to=TensorProto.FLOAT),

            # 1ch → 3ch
            helper.make_node("Expand", ["hit_f", "expand_shape"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
            initializer=[luma_init, axes_init, fg_init, bg_init,
                         fg_sum_t, bg_sum_t, one, threshold, expand_init],
        )
