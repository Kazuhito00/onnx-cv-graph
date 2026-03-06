"""Extended Difference of Gaussians (XDoG) スケッチフィルタの ONNX グラフ定義."""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec

from .gaussian_blur import _gaussian_kernel_2d


class XDoGOp(OnnxGraphOp):
    """Extended Difference of Gaussians (XDoG) スケッチフィルタ.

    2つのガウシアンぼかしの差分 (DoG) に Tanh 軟閾値を適用し、
    白黒スケッチ風の画像を生成する。

    アルゴリズム:
        gray     = luma(input)
        g1       = GaussianBlur(gray, k1)
        g2       = GaussianBlur(gray, k2)
        dog      = g1 - γ × g2
        dog_norm = dog / max(ReduceMax(dog), ε_safe)
        e        = 1 + tanh(φ × (dog_norm − ε))
        output   = clip(e, 0, 1) → 3ch 拡張

    パラメータ (グラフ定義時固定):
        k1 (int):    σ小 のガウシアンカーネルサイズ (奇数)
        k2 (int):    σ大 のガウシアンカーネルサイズ (奇数, k2 > k1)
        gamma (float): DoG 抑制係数 γ (典型値 0.98)
        phi (float):   Tanh シャープネス係数 φ (典型値 200)
        eps (float):   軟閾値オフセット ε (典型値 -0.1)

    ノード構成:
        Mul+ReduceSum (グレースケール) → Pad+Conv×2 (2段 Gaussian) →
        Mul+Sub (DoG) → ReduceMax+Max+Div (正規化) →
        Sub+Mul+Tanh+Add (XDoG) → Clip → Expand (3ch)
    """

    def __init__(
        self,
        k1: int = 3,
        k2: int = 5,
        gamma: float = 0.98,
        phi: float = 200.0,
        eps: float = -0.1,
    ):
        assert k1 % 2 == 1 and k2 % 2 == 1 and k1 >= 3 and k2 > k1, \
            "k1・k2 は奇数の正の整数、かつ k2 > k1"
        self._k1 = k1
        self._k2 = k2
        self._gamma = gamma
        self._phi = phi
        self._eps = eps

    @property
    def op_name(self) -> str:
        return f"xdog_{self._k1}x{self._k1}_{self._k2}x{self._k2}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        return [cls(3, 5), cls(5, 9), cls(7, 13)]

    def build_graph(self) -> GraphProto:
        k1, k2 = self._k1, self._k2
        pad1, pad2 = k1 // 2, k2 // 2

        # --- 初期化テンソル ---
        luma = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        luma_init = numpy_helper.from_array(luma, "luma")

        axes1_init = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes1")

        g1_kernel = _gaussian_kernel_2d(k1).reshape(1, 1, k1, k1)
        g2_kernel = _gaussian_kernel_2d(k2).reshape(1, 1, k2, k2)
        kernel1_init = numpy_helper.from_array(g1_kernel, "kernel1")
        kernel2_init = numpy_helper.from_array(g2_kernel, "kernel2")

        pads1 = np.array([0, 0, pad1, pad1, 0, 0, pad1, pad1], dtype=np.int64)
        pads1_init = numpy_helper.from_array(pads1, "pads1")
        pads2 = np.array([0, 0, pad2, pad2, 0, 0, pad2, pad2], dtype=np.int64)
        pads2_init = numpy_helper.from_array(pads2, "pads2")

        gamma_init = numpy_helper.from_array(np.float32(self._gamma), "gamma")
        phi_init = numpy_helper.from_array(np.float32(self._phi), "phi")
        eps_init = numpy_helper.from_array(np.float32(self._eps), "eps")
        safe_eps_init = numpy_helper.from_array(np.float32(1e-6), "safe_eps")
        one_init = numpy_helper.from_array(np.float32(1.0), "one")
        zero_init = numpy_helper.from_array(np.float32(0.0), "zero")
        expand_shape_init = numpy_helper.from_array(
            np.array([1, 3, 1, 1], dtype=np.int64), "expand_shape"
        )

        initializers = [
            luma_init, axes1_init,
            kernel1_init, kernel2_init,
            pads1_init, pads2_init,
            gamma_init, phi_init, eps_init,
            safe_eps_init, one_init, zero_init,
            expand_shape_init,
        ]

        nodes = [
            # 1. グレースケール変換 (N,3,H,W) → (N,1,H,W)
            helper.make_node("Mul", ["input", "luma"], ["weighted"]),
            helper.make_node("ReduceSum", ["weighted", "axes1"], ["gray"], keepdims=1),

            # 2. σ1 ガウシアンぼかし (Pad → Conv)
            helper.make_node("Pad", ["gray", "pads1"], ["padded1"], mode="reflect"),
            helper.make_node("Conv", ["padded1", "kernel1"], ["g1"]),

            # 3. σ2 ガウシアンぼかし (Pad → Conv)
            helper.make_node("Pad", ["gray", "pads2"], ["padded2"], mode="reflect"),
            helper.make_node("Conv", ["padded2", "kernel2"], ["g2"]),

            # 4. DoG = g1 - γ × g2
            helper.make_node("Mul", ["g2", "gamma"], ["g2_scaled"]),
            helper.make_node("Sub", ["g1", "g2_scaled"], ["dog"]),

            # 5. 正規化: dog / max(ReduceMax(dog, H,W), safe_eps)
            helper.make_node("ReduceMax", ["dog"], ["dog_max"], axes=[2, 3], keepdims=1),
            helper.make_node("Max", ["dog_max", "safe_eps"], ["dog_max_safe"]),
            helper.make_node("Div", ["dog", "dog_max_safe"], ["dog_norm"]),

            # 6. XDoG: 1 + tanh(φ × (dog_norm - ε))
            helper.make_node("Sub", ["dog_norm", "eps"], ["shifted"]),
            helper.make_node("Mul", ["shifted", "phi"], ["phi_scaled"]),
            helper.make_node("Tanh", ["phi_scaled"], ["tanh_val"]),
            helper.make_node("Add", ["one", "tanh_val"], ["e"]),

            # 7. Clip [0, 1] → 3ch 拡張
            helper.make_node("Clip", ["e", "zero", "one"], ["clipped"]),
            helper.make_node("Expand", ["clipped", "expand_shape"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi], [output_vi],
            initializer=initializers,
        )
