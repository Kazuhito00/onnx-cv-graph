"""ImageNet 正規化モデルのテスト."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "imagenet_norm.onnx"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    if not MODEL_PATH.exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )
    assert MODEL_PATH.exists()


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_PATH))


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


class TestImageNetNormOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (2, 3, 8, 8)


class TestImageNetNormValues:
    def test_mean_image_to_zero(self, session):
        """mean と同じ画像を入力すると出力が 0 になること."""
        img = np.tile(IMAGENET_MEAN, (1, 1, 4, 4))
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_matches_numpy(self, session):
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 8, 8), dtype=np.float32)
        expected = (img - IMAGENET_MEAN) / IMAGENET_STD
        out = _run(session, img)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_output_range_exceeds_01(self, session):
        """出力が [0,1] 範囲外に出ること (ML ドメインの確認)."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img)
        # (0 - 0.485) / 0.229 ≈ -2.12 なので 0 未満になる
        assert out.min() < 0.0

