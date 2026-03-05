"""[0,1]→[0,255] スケーリングモデルのテスト."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "scale_to_255.onnx"


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


class TestScaleTo255OutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (2, 3, 8, 8)


class TestScaleTo255Values:
    def test_zero_to_zero(self, session):
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_one_to_255(self, session):
        img = np.ones((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 255.0, atol=1e-3)

    def test_half_to_127_5(self, session):
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 127.5, atol=1e-3)

    def test_matches_numpy(self, session):
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, img * 255.0, atol=1e-3)


class TestScaleTo255Domain:
    def test_output_domain(self):
        from src.onnx_cv_graph import ScaleTo255Op
        op = ScaleTo255Op()
        assert op.input_domain == "image"
        assert op.output_domain == "ml"
