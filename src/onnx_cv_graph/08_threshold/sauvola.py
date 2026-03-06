"""Sauvola 局所適応2値化の ONNX グラフ定義.

ドキュメント画像の局所平均・局所標準偏差を用いた適応的2値化。
Niblack の改良版で、照明ムラや背景のグラデーションに強い。

論文: J. Sauvola & M. Pietikainen, "Adaptive document image binarization" (2000)

閾値式: T = μ × [1 + k × (σ/R − 1)]
  μ: 局所平均 (AveragePool)
  σ: 局所標準偏差 (√(E[X²] − μ²))
  R: 標準偏差の最大理論値 (= 0.5、[0,1] 正規化画像の場合)
  k: 感度パラメータ (推論時入力、デフォルト 0.5)

block_size はグラフ定義時に固定。OCR 用途では 15/31/63 の3種を生成する。
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class SauvolaOp(OnnxGraphOp):
    """Sauvola 局所適応2値化.

    グレースケール化 → AveragePool×2 (局所平均・E[X²]) →
    分散/標準偏差 → 閾値 T = μ×(1+k×(σ/R−1)) → Greater → 3ch 拡張.

    ノード数: 18 (★★★★)
    """

    def __init__(self, kernel_size: int = 15):
        assert kernel_size % 2 == 1 and kernel_size >= 3, \
            "kernel_size は奇数かつ 3 以上"
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        k = self._kernel_size
        return f"sauvola_{k}x{k}"

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
        return {"k": (0.0, 1.0, 0.5)}

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        return [cls(15), cls(31), cls(63)]

    def build_graph(self) -> GraphProto:
        ks = self._kernel_size
        pad = ks // 2

        # --- 初期化テンソル ---
        luma_w = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        luma_w_init = numpy_helper.from_array(luma_w, "luma_w")

        axes1_init = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes1")

        # Pad: [top, left, bottom, right] × [N, C] → ONNX Pad 形式 [N,C,H,W 各始端/終端]
        pads_arr = np.array([0, 0, pad, pad, 0, 0, pad, pad], dtype=np.int64)
        pads_init = numpy_helper.from_array(pads_arr, "pads")

        # R = 0.5: [0,1] 正規化画像における標準偏差の最大値
        R_init = numpy_helper.from_array(np.float32(0.5), "R")
        one_init = numpy_helper.from_array(np.float32(1.0), "one")
        zero_init = numpy_helper.from_array(np.float32(0.0), "zero")
        expand_shape_init = numpy_helper.from_array(
            np.array([1, 3, 1, 1], dtype=np.int64), "expand_shape"
        )

        initializers = [
            luma_w_init, axes1_init, pads_init,
            R_init, one_init, zero_init, expand_shape_init,
        ]

        nodes = [
            # 1. グレースケール変換 (N,3,H,W) → (N,1,H,W)
            helper.make_node("Mul", ["input", "luma_w"], ["weighted"]),
            helper.make_node("ReduceSum", ["weighted", "axes1"], ["gray"], keepdims=1),

            # 2. 輝度の自乗 E[X²] 計算用
            helper.make_node("Mul", ["gray", "gray"], ["gray2"]),

            # 3. 局所平均 μ: reflect padding → AveragePool
            helper.make_node("Pad", ["gray", "pads"], ["gray_pad"], mode="reflect"),
            helper.make_node(
                "AveragePool", ["gray_pad"], ["mu"],
                kernel_shape=[ks, ks], strides=[1, 1],
            ),

            # 4. 局所 E[X²]: reflect padding → AveragePool
            helper.make_node("Pad", ["gray2", "pads"], ["gray2_pad"], mode="reflect"),
            helper.make_node(
                "AveragePool", ["gray2_pad"], ["e_x2"],
                kernel_shape=[ks, ks], strides=[1, 1],
            ),

            # 5. 局所分散 σ² = E[X²] − μ²
            helper.make_node("Mul", ["mu", "mu"], ["mu_sq"]),
            helper.make_node("Sub", ["e_x2", "mu_sq"], ["var_raw"]),
            helper.make_node("Max", ["var_raw", "zero"], ["var"]),   # 浮動小数誤差で負になることを防止

            # 6. 局所標準偏差 σ = √var
            helper.make_node("Sqrt", ["var"], ["sigma"]),

            # 7. 閾値計算: T = μ × [1 + k × (σ/R − 1)]
            helper.make_node("Div", ["sigma", "R"], ["sigma_norm"]),       # σ/R
            helper.make_node("Sub", ["sigma_norm", "one"], ["sigma_m1"]),  # σ/R − 1
            helper.make_node("Mul", ["sigma_m1", "k"], ["k_term"]),        # k × (σ/R − 1)
            helper.make_node("Add", ["one", "k_term"], ["scale"]),         # 1 + k × (σ/R − 1)
            helper.make_node("Mul", ["mu", "scale"], ["threshold"]),       # T

            # 8. 2値化: gray > T → float → 3ch 拡張
            helper.make_node("Greater", ["gray", "threshold"], ["mask"]),
            helper.make_node("Cast", ["mask"], ["bin_1ch"], to=TensorProto.FLOAT),
            helper.make_node("Expand", ["bin_1ch", "expand_shape"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        k_vi = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name,
            [input_vi, k_vi],
            [output_vi],
            initializer=initializers,
        )
