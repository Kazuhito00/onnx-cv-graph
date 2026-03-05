"""RGB→BGR チャネルスワップモデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "rgb2bgr.onnx"


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


class TestRgb2BgrOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (2, 3, 8, 8)


class TestRgb2BgrValues:
    def test_channels_swapped(self, session):
        """R↔B が入れ替わり、G はそのままであること."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_array_equal(out[:, 0], img[:, 2])  # B←R
        np.testing.assert_array_equal(out[:, 1], img[:, 1])  # G←G
        np.testing.assert_array_equal(out[:, 2], img[:, 0])  # R←B

    def test_double_swap_identity(self, session):
        """2回適用で元に戻ること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out1 = _run(session, img)
        out2 = _run(session, out1)
        np.testing.assert_array_equal(out2, img)

    def test_pure_red(self, session):
        """純赤 (1,0,0) → (0,0,1) になること."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        img[:, 0] = 1.0
        out = _run(session, img)
        np.testing.assert_array_equal(out[:, 0], 0.0)
        np.testing.assert_array_equal(out[:, 1], 0.0)
        np.testing.assert_array_equal(out[:, 2], 1.0)
