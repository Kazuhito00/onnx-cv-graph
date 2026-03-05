"""ChainOp のドメイン互換性検証テスト."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import onnxruntime as ort
import pytest

from src.chain import ChainOp
from src.onnx_cv_graph import (
    GrayscaleOp,
    ImageNetNormOp,
    ScaleFrom255Op,
    ScaleTo255Op,
)

MODEL_DIR = PROJECT_ROOT / "models"


class TestChainDomainValidation:
    def test_image_to_ml_allowed(self):
        """image → ml のチェーンは許可される."""
        chain = ChainOp([GrayscaleOp(), ImageNetNormOp()])
        assert chain.input_domain == "image"
        assert chain.output_domain == "ml"

    def test_ml_to_image_allowed(self):
        """image→ml → ml→image のチェーンは許可される."""
        chain = ChainOp([ScaleTo255Op(), ScaleFrom255Op()])
        assert chain.input_domain == "image"
        assert chain.output_domain == "image"

    def test_image_after_ml_rejected(self):
        """ml 出力の後に image 入力の op を接続するとエラー."""
        with pytest.raises(ValueError, match="ドメイン不一致"):
            ChainOp([ImageNetNormOp(), GrayscaleOp()])

    def test_ml_input_after_image_rejected(self):
        """image 出力の後に ml 入力の op を接続するとエラー."""
        with pytest.raises(ValueError, match="ドメイン不一致"):
            ChainOp([GrayscaleOp(), ScaleFrom255Op()])


class TestChainDomainExecution:
    def test_scale_roundtrip(self):
        """scale_to_255 → scale_from_255 で値が元に戻ること."""
        chain = ChainOp([ScaleTo255Op(), ScaleFrom255Op()])
        path = MODEL_DIR / "_test_scale_roundtrip.onnx"
        try:
            chain.export(path)
            sess = ort.InferenceSession(str(path))
            rng = np.random.default_rng(42)
            img = rng.random((1, 3, 8, 8), dtype=np.float32)
            out = sess.run(None, {"input": img})[0]
            np.testing.assert_allclose(out, img, atol=1e-5)
        finally:
            if path.exists():
                path.unlink()

    def test_grayscale_then_imagenet(self):
        """grayscale → imagenet_norm のチェーンが実行できること."""
        chain = ChainOp([GrayscaleOp(), ImageNetNormOp()])
        path = MODEL_DIR / "_test_gray_imagenet.onnx"
        try:
            chain.export(path)
            sess = ort.InferenceSession(str(path))
            img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
            out = sess.run(None, {"input": img})[0]
            # グレースケール後の値 ≈ 0.5 (各ch同一) → ImageNet 正規化
            assert out.shape == (1, 3, 4, 4)
        finally:
            if path.exists():
                path.unlink()
