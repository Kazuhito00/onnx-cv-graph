"""[0,255]→[0,1] スケーリングモデルのテスト."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "scale_from_255.onnx"


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


class TestScaleFrom255OutputShape:
    def test_single_image(self, session):
        img = (np.random.rand(1, 3, 8, 8) * 255).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 8, 8)


class TestScaleFrom255Values:
    def test_zero_to_zero(self, session):
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_255_to_one(self, session):
        img = np.full((1, 3, 4, 4), 255.0, dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 1.0, atol=1e-5)

    def test_roundtrip(self, session):
        """scale_to_255 → scale_from_255 で元に戻ること."""
        to_path = PROJECT_ROOT / "models" / CATEGORY / "scale_to_255.onnx"
        to_sess = ort.InferenceSession(str(to_path))
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        scaled = to_sess.run(None, {"input": img})[0]
        restored = _run(session, scaled)
        np.testing.assert_allclose(restored, img, atol=1e-5)

