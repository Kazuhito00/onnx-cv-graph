"""[-1,1] 正規化モデルのテスト."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "normalize_neg1_pos1.onnx"


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


class TestNormalizeNeg1Pos1OutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 8, 8)


class TestNormalizeNeg1Pos1Values:
    def test_zero_to_neg1(self, session):
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, -1.0, atol=1e-5)

    def test_one_to_pos1(self, session):
        img = np.ones((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 1.0, atol=1e-5)

    def test_half_to_zero(self, session):
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_matches_numpy(self, session):
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, img * 2.0 - 1.0, atol=1e-5)

