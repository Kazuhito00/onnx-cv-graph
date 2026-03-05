"""バッチ次元除去モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "batch_squeeze.onnx"


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


class TestBatchSqueezeOutputShape:
    def test_removes_batch_dim(self, session):
        """(1, 3, H, W) → (3, H, W)."""
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (3, 8, 8)

    def test_various_sizes(self, session):
        img = np.random.rand(1, 3, 16, 12).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (3, 16, 12)


class TestBatchSqueezeValues:
    def test_values_preserved(self, session):
        """値が変わらない."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, img[0], atol=1e-6)
