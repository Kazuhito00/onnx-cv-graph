"""Prewitt エッジ検出モデルのテスト."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "prewitt.onnx"


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


class TestPrewittOutputShape:
    def test_single_image(self, session):
        assert _run(session, np.random.rand(1, 3, 8, 8).astype(np.float32)).shape == (1, 3, 8, 8)

    def test_batch(self, session):
        assert _run(session, np.random.rand(2, 3, 8, 8).astype(np.float32)).shape == (2, 3, 8, 8)


class TestPrewittValues:
    def test_uniform_is_zero(self, session):
        img = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_output_range(self, session):
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_3ch_identical(self, session):
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])

    def test_edge_detected(self, session):
        img = np.zeros((1, 3, 8, 8), dtype=np.float32)
        img[:, :, :, 4:] = 1.0
        out = _run(session, img)
        assert out[0, 0, :, 3:5].mean() > out[0, 0, :, 0:2].mean()
