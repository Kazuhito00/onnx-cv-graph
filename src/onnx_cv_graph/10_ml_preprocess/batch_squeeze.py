"""バッチ次元除去の ONNX グラフ定義.

NCHW 版: バッチ付き (1, 3, H, W) → 単一画像 (3, H, W)
NHWC 版: バッチ付き (1, H, W, 3) → 単一画像 (H, W, 3)

どちらも Squeeze(axis=0) で実現する。
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class BatchSqueezeNchwOp(OnnxGraphOp):
    """バッチ次元除去 (NCHW): (1,3,H,W) → (3,H,W) = Squeeze(axis=0)."""

    @property
    def op_name(self) -> str:
        return "batch_squeeze_nchw"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, [1, 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, [3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes")
        node = helper.make_node("Squeeze", ["input", "axes"], ["output"])
        input_vi  = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, "H", "W"])
        return helper.make_graph([node], self.op_name, [input_vi], [output_vi], initializer=[axes])


class BatchSqueezeNhwcOp(OnnxGraphOp):
    """バッチ次元除去 (NHWC): (1,H,W,3) → (H,W,3) = Squeeze(axis=0).

    chw_to_hwc の出力からバッチ次元を除いて OpenCV に渡す用途に使う。
    """

    @property
    def op_name(self) -> str:
        return "batch_squeeze_nhwc"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, [1, "H", "W", 3])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["H", "W", 3])]

    def build_graph(self) -> GraphProto:
        axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes")
        node = helper.make_node("Squeeze", ["input", "axes"], ["output"])
        input_vi  = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, "H", "W", 3])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["H", "W", 3])
        return helper.make_graph([node], self.op_name, [input_vi], [output_vi], initializer=[axes])
