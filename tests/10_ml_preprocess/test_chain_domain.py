"""ChainOp の合成実行テスト."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import onnxruntime as ort

from src.chain import ChainOp
from src.onnx_cv_graph import (
    GrayscaleOp,
    ImageNetNormOp,
    ScaleFrom255Op,
    ScaleTo255Op,
)

MODEL_DIR = PROJECT_ROOT / "models"


class TestChainExecution:
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
            assert out.shape == (1, 3, 4, 4)
        finally:
            if path.exists():
                path.unlink()
