"""局所コントラスト正規化 (Local Contrast Normalization, LCN) の ONNX グラフ定義.

各ピクセルから局所平均を引き、局所標準偏差で割ることで
照明ムラ・背景グラデーション・局所コントラスト差を除去する。

式:
  μ(x,y)    = AveragePool(input)          ← 局所平均 (per-channel)
  σ(x,y)    = √(E[X²] − μ²)              ← 局所標準偏差
  normalized = (input − μ) / (σ + ε)
  output     = Sigmoid(normalized)         ← (0,1) に写像

出力の解釈:
  - 0.5 付近: 局所平均値 (背景)
  - > 0.5  : 周囲より明るい領域
  - < 0.5  : 周囲より暗い領域 (文字など)

OCR 前処理として BgNormalize (単純差分) より高精度な照明ムラ除去が可能。
block_size はグラフ定義時に固定。15/31/63 の3種を生成する。
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class LcnOp(OnnxGraphOp):
    """局所コントラスト正規化 (LCN).

    AveragePool×2 (局所平均・E[X²]) → 分散/標準偏差 →
    (input − μ) / (σ + ε) → Sigmoid → (0,1).

    各 RGB チャネルに独立に適用する。
    ノード数: 12 (★★★)
    """

    def __init__(self, kernel_size: int = 15):
        assert kernel_size % 2 == 1 and kernel_size >= 3, \
            "kernel_size は奇数かつ 3 以上"
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        k = self._kernel_size
        return f"lcn_{k}x{k}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        return [cls(15), cls(31)]

    def build_graph(self) -> GraphProto:
        ks = self._kernel_size
        pad = ks // 2

        # --- 初期化テンソル ---
        pads_arr = np.array([0, 0, pad, pad, 0, 0, pad, pad], dtype=np.int64)
        pads_init = numpy_helper.from_array(pads_arr, "pads")

        eps_init = numpy_helper.from_array(np.float32(1e-5), "eps")
        zero_init = numpy_helper.from_array(np.float32(0.0), "zero")

        initializers = [pads_init, eps_init, zero_init]

        nodes = [
            # 1. 入力の自乗 E[X²] 計算用
            helper.make_node("Mul", ["input", "input"], ["input2"]),

            # 2. 局所平均 μ (per-channel): reflect padding → AveragePool
            helper.make_node("Pad", ["input", "pads"], ["input_pad"], mode="reflect"),
            helper.make_node(
                "AveragePool", ["input_pad"], ["mu"],
                kernel_shape=[ks, ks], strides=[1, 1],
            ),

            # 3. 局所 E[X²]: reflect padding → AveragePool
            helper.make_node("Pad", ["input2", "pads"], ["input2_pad"], mode="reflect"),
            helper.make_node(
                "AveragePool", ["input2_pad"], ["e_x2"],
                kernel_shape=[ks, ks], strides=[1, 1],
            ),

            # 4. 局所分散 σ² = E[X²] − μ²
            helper.make_node("Mul", ["mu", "mu"], ["mu_sq"]),
            helper.make_node("Sub", ["e_x2", "mu_sq"], ["var_raw"]),
            helper.make_node("Max", ["var_raw", "zero"], ["var"]),   # 浮動小数誤差で負になることを防止

            # 5. 局所標準偏差 σ + ε (ゼロ除算防止)
            helper.make_node("Sqrt", ["var"], ["sigma"]),
            helper.make_node("Add", ["sigma", "eps"], ["sigma_safe"]),

            # 6. 正規化: (input − μ) / (σ + ε)
            helper.make_node("Sub", ["input", "mu"], ["centered"]),
            helper.make_node("Div", ["centered", "sigma_safe"], ["normalized"]),

            # 7. Sigmoid → (0, 1) に写像
            helper.make_node("Sigmoid", ["normalized"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name,
            [input_vi],
            [output_vi],
            initializer=initializers,
        )
