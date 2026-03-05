"""アンシャープマスクの ONNX グラフ定義.

output = input + amount * (input - GaussianBlur(input))
        = input * (1 + amount) - GaussianBlur(input) * amount → Clip(0,1)

amount パラメータでシャープ強度を制御する.
"""

from typing import Dict, List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec
from .gaussian_blur import _gaussian_kernel_2d


class UnsharpMaskOp(OnnxGraphOp):
    """アンシャープマスク.

    Pad → Conv(Gaussian) → Sub(input - blurred) → Mul(amount) → Add(input) → Clip.
    """

    def __init__(self, kernel_size: int = 3):
        self._kernel_size = kernel_size

    @property
    def op_name(self) -> str:
        return f"unsharp_mask_{self._kernel_size}x{self._kernel_size}"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [
            ("input", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            ("amount", TensorProto.FLOAT, [1]),
        ]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        return {"amount": (0.0, 5.0, 1.0)}

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        return [cls(3), cls(5), cls(7)]

    def build_graph(self) -> GraphProto:
        k = self._kernel_size
        pad = k // 2

        g2d = _gaussian_kernel_2d(k)
        kernel = np.stack([g2d] * 3).reshape(3, 1, k, k)
        kernel_init = numpy_helper.from_array(kernel, name="kernel")

        pads = np.array([0, 0, pad, pad, 0, 0, pad, pad], dtype=np.int64)
        pads_init = numpy_helper.from_array(pads, name="pads")
        zero = np.array([0.0], dtype=np.float32)
        zero_init = numpy_helper.from_array(zero, name="zero")
        one = np.array([1.0], dtype=np.float32)
        one_init = numpy_helper.from_array(one, name="one")

        nodes = [
            # ガウシアンぼかし
            helper.make_node("Pad", ["input", "pads"], ["padded"], mode="reflect"),
            helper.make_node("Conv", ["padded", "kernel"], ["blurred"], group=3),
            # detail = input - blurred
            helper.make_node("Sub", ["input", "blurred"], ["detail"]),
            # detail * amount
            helper.make_node("Mul", ["detail", "amount"], ["scaled_detail"]),
            # input + scaled_detail
            helper.make_node("Add", ["input", "scaled_detail"], ["enhanced"]),
            # Clip(0, 1)
            helper.make_node("Clip", ["enhanced", "zero", "one"], ["output"]),
        ]

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        amount_vi = helper.make_tensor_value_info("amount", TensorProto.FLOAT, [1])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            nodes, self.op_name, [input_vi, amount_vi], [output_vi],
            initializer=[kernel_init, pads_init, zero_init, one_init],
        )
