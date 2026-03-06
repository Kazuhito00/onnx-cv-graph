"""ピクセル平均減算 (Caffe/detectron2 系) モデルのテスト."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "pixel_mean_sub.onnx"

PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 3, 1, 1)


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


class TestPixelMeanSubOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (2, 3, 8, 8)


class TestPixelMeanSubValues:
    def test_matches_numpy(self, session):
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 8, 8), dtype=np.float32)
        expected = img * 255.0 - PIXEL_MEAN
        out = _run(session, img)
        np.testing.assert_allclose(out, expected, atol=1e-3)

    def test_zero_input(self, session):
        """入力 0 → -mean が出力される."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out[:, 0], -123.675, atol=1e-3)
        np.testing.assert_allclose(out[:, 1], -116.28, atol=1e-3)
        np.testing.assert_allclose(out[:, 2], -103.53, atol=1e-3)

    def test_mean_input_to_zero(self, session):
        """mean / 255 を入力すると出力がほぼ 0."""
        img = PIXEL_MEAN / 255.0
        img = np.tile(img, (1, 1, 4, 4))
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-3)

