"""バッチ次元追加の ONNX グラフ定義.

NCHW 版: 単一画像 (3, H, W)   → バッチ付き (1, 3, H, W)
NHWC 版: 単一画像 (H, W, 3)   → バッチ付き (1, H, W, 3)

どちらも Unsqueeze(axis=0) で実現する。
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class BatchUnsqueezeNchwOp(OnnxGraphOp):
    """バッチ次元追加 (NCHW): (3,H,W) → (1,3,H,W) = Unsqueeze(axis=0)."""

    @property
    def op_name(self) -> str:
        return "batch_unsqueeze_nchw"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, [3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, [1, 3, "H", "W"])]

    def build_graph(self) -> GraphProto:
        axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes")
        node = helper.make_node("Unsqueeze", ["input", "axes"], ["output"])
        input_vi  = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "H", "W"])
        return helper.make_graph([node], self.op_name, [input_vi], [output_vi], initializer=[axes])


class BatchUnsqueezeNhwcOp(OnnxGraphOp):
    """バッチ次元追加 (NHWC): (H,W,3) → (1,H,W,3) = Unsqueeze(axis=0).

    OpenCV で読み込んだ HWC 画像にバッチ次元を付けて hwc_to_chw に渡す用途に使う。
    """

    @property
    def op_name(self) -> str:
        return "batch_unsqueeze_nhwc"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["H", "W", 3])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, [1, "H", "W", 3])]

    def build_graph(self) -> GraphProto:
        axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes")
        node = helper.make_node("Unsqueeze", ["input", "axes"], ["output"])
        input_vi  = helper.make_tensor_value_info("input",  TensorProto.FLOAT, ["H", "W", 3])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, "H", "W", 3])
        return helper.make_graph([node], self.op_name, [input_vi], [output_vi], initializer=[axes])
