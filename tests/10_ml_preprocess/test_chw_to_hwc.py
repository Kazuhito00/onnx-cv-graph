"""CHW→HWC 変換モデルのテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "chw_to_hwc.onnx"


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


def _run(session, img):
    return session.run(None, {"input": img})[0]


class TestChwToHwcOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 8, 8, 3)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 16, 12).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (2, 16, 12, 3)


class TestChwToHwcValues:
    def test_values_preserved(self, session):
        """値が変わらず軸のみ転置."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        expected = img.transpose(0, 2, 3, 1)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_channel_order(self, session):
        """各チャネルが正しい位置に転置される."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        img[0, 0] = 0.1  # R
        img[0, 1] = 0.5  # G
        img[0, 2] = 0.9  # B
        out = _run(session, img)
        np.testing.assert_allclose(out[0, :, :, 0], 0.1, atol=1e-6)
        np.testing.assert_allclose(out[0, :, :, 1], 0.5, atol=1e-6)
        np.testing.assert_allclose(out[0, :, :, 2], 0.9, atol=1e-6)


class TestChwToHwcDomain:
    def test_domain(self):
        from src.onnx_cv_graph import ChwToHwcOp
        op = ChwToHwcOp()
        assert op.input_domain == "image"
        assert op.output_domain == "ml"
