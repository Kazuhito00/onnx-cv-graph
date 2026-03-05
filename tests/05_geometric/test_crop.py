"""クロップ (crop) モデルのテスト.

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
MODEL_DIR = PROJECT_ROOT / "models" / CATEGORY


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    if not (MODEL_DIR / "crop.onnx").exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_DIR / "crop.onnx"))


def _run(session, img: np.ndarray, top=0.0, left=0.0, h=1.0, w=1.0) -> np.ndarray:
    return session.run(None, {
        "input": img,
        "crop_top": np.array([top], dtype=np.float32),
        "crop_left": np.array([left], dtype=np.float32),
        "crop_h": np.array([h], dtype=np.float32),
        "crop_w": np.array([w], dtype=np.float32),
    })[0]


class TestCropOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_full_crop(self, session):
        """top=0, left=0, h=1, w=1 で全体がそのまま返ること."""
        img = np.random.rand(1, 3, 16, 20).astype(np.float32)
        out = _run(session, img, 0.0, 0.0, 1.0, 1.0)
        assert out.shape == (1, 3, 16, 20)

    def test_half_crop(self, session):
        """h=0.5, w=0.5 で半分サイズが返ること."""
        img = np.random.rand(1, 3, 16, 20).astype(np.float32)
        out = _run(session, img, 0.0, 0.0, 0.5, 0.5)
        assert out.shape == (1, 3, 8, 10)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 16, 20).astype(np.float32)
        out = _run(session, img, 0.25, 0.25, 0.5, 0.5)
        assert out.shape[0] == 2
        assert out.shape[1] == 3


class TestCropValues:
    """出力値の正確性を検証するテスト群."""

    def test_full_crop_identity(self, session):
        """全体クロップで入力と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 20), dtype=np.float32)
        out = _run(session, img, 0.0, 0.0, 1.0, 1.0)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_top_left_quarter(self, session):
        """左上1/4 のクロップが正しい領域を返すこと."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 20), dtype=np.float32)
        out = _run(session, img, 0.0, 0.0, 0.5, 0.5)
        expected = img[:, :, :8, :10]
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_bottom_right_quarter(self, session):
        """右下1/4 のクロップが正しい領域を返すこと."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 20), dtype=np.float32)
        out = _run(session, img, 0.5, 0.5, 0.5, 0.5)
        expected = img[:, :, 8:16, 10:20]
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_uniform_image(self, session):
        """均一画像はクロップ後も同一値であること."""
        img = np.full((1, 3, 16, 16), 0.7, dtype=np.float32)
        out = _run(session, img, 0.25, 0.25, 0.5, 0.5)
        np.testing.assert_allclose(out, 0.7, atol=1e-5)
