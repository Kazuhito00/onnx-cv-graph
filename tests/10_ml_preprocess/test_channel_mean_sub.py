"""チャネルごと平均減算モデルのテスト."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "channel_mean_sub.onnx"

DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)


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


class TestChannelMeanSubOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 8, 8)


class TestChannelMeanSubValues:
    def test_mean_image_to_zero(self, session):
        """mean と同じ画像を入力すると出力が 0."""
        img = np.tile(DEFAULT_MEAN, (1, 1, 4, 4))
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_matches_numpy(self, session):
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 8, 8), dtype=np.float32)
        expected = img - DEFAULT_MEAN
        out = _run(session, img)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_output_can_be_negative(self, session):
        """出力が負になりうること (ML ドメイン)."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img)
        assert out.min() < 0.0


class TestChannelMeanSubDomain:
    def test_output_domain(self):
        from src.onnx_cv_graph import ChannelMeanSubOp
        op = ChannelMeanSubOp()
        assert op.input_domain == "image"
        assert op.output_domain == "ml"
