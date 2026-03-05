"""エンボスモデルのテスト."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "emboss.onnx"


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


class TestEmbossOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        assert _run(session, img).shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        assert _run(session, img).shape == (2, 3, 8, 8)


class TestEmbossValues:
    def test_uniform_preserves_value(self, session):
        """均一画像のエンボス: カーネル合計=1 なので val*1+0.5."""
        img = np.full((1, 3, 8, 8), 0.3, dtype=np.float32)
        out = _run(session, img)
        # カーネル合計 = -2-1+0-1+1+1+0+1+2 = 1
        # 均一画像: conv = 0.3*1 = 0.3, +0.5 = 0.8
        np.testing.assert_allclose(out, 0.8, atol=0.02)

    def test_output_range(self, session):
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_matches_numpy(self, session):
        correlate = pytest.importorskip("scipy.ndimage").correlate
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        k = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        expected = np.zeros_like(img)
        for n in range(img.shape[0]):
            for c in range(img.shape[1]):
                expected[n, c] = correlate(img[n, c], k, mode='mirror')
        expected = np.clip(expected + 0.5, 0, 1)
        np.testing.assert_allclose(out, expected, atol=1e-5)
