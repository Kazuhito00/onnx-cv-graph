"""ImageNet 正規化の ONNX グラフ定義.

torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 相当.
画像ドメイン → ML ドメイン.
"""

from typing import List

import numpy as np
from onnx import GraphProto, TensorProto, helper, numpy_helper

from src.base import OnnxGraphOp, TensorSpec


class ImageNetNormOp(OnnxGraphOp):
    """ImageNet 正規化.

    入力 [0,1] (RGB) から ImageNet の mean/std で正規化する.
    output = (input - mean) / std
    """

    @property
    def op_name(self) -> str:
        return "imagenet_norm"

    @property
    def input_specs(self) -> List[TensorSpec]:
        return [("input", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_specs(self) -> List[TensorSpec]:
        return [("output", TensorProto.FLOAT, ["N", 3, "H", "W"])]

    @property
    def output_domain(self) -> str:
        return "ml"

    def build_graph(self) -> GraphProto:
        # ImageNet mean/std (1, 3, 1, 1) でブロードキャスト
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        mean_init = numpy_helper.from_array(mean, name="mean")
        std_init = numpy_helper.from_array(std, name="std")

        sub_node = helper.make_node("Sub", ["input", "mean"], ["centered"])
        div_node = helper.make_node("Div", ["centered", "std"], ["output"])

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])

        return helper.make_graph(
            [sub_node, div_node],
            self.op_name,
            [input_vi],
            [output_vi],
            initializer=[mean_init, std_init],
        )
