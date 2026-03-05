"""uint8→float32 変換モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "uint8_to_float.onnx"


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


class TestUint8ToFloatOutputShape:
    def test_single_image(self, session):
        img = np.random.randint(0, 256, (1, 3, 8, 8), dtype=np.uint8)
        out = _run(session, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.randint(0, 256, (2, 3, 8, 8), dtype=np.uint8)
        out = _run(session, img)
        assert out.shape == (2, 3, 8, 8)


class TestUint8ToFloatDtype:
    def test_output_is_float32(self, session):
        img = np.random.randint(0, 256, (1, 3, 4, 4), dtype=np.uint8)
        out = _run(session, img)
        assert out.dtype == np.float32


class TestUint8ToFloatValues:
    def test_zero_to_zero(self, session):
        img = np.zeros((1, 3, 4, 4), dtype=np.uint8)
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_255_to_one(self, session):
        img = np.full((1, 3, 4, 4), 255, dtype=np.uint8)
        out = _run(session, img)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_128_to_half(self, session):
        img = np.full((1, 3, 4, 4), 128, dtype=np.uint8)
        out = _run(session, img)
        np.testing.assert_allclose(out, 128.0 / 255.0, atol=1e-6)

    def test_matches_numpy(self, session):
        """NumPy 相当の変換と一致."""
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (1, 3, 8, 8), dtype=np.uint8)
        out = _run(session, img)
        expected = img.astype(np.float32) / 255.0
        np.testing.assert_allclose(out, expected, atol=1e-6)


class TestUint8ToFloatDomain:
    def test_domain(self):
        from src.onnx_cv_graph import Uint8ToFloatOp
        op = Uint8ToFloatOp()
        assert op.input_domain == "ml"
        assert op.output_domain == "image"
