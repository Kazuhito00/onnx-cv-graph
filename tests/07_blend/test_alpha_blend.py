"""アルファブレンドモデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "alpha_blend.onnx"


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    if not MODEL_PATH.exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )
    assert MODEL_PATH.exists(), f"モデルが見つかりません: {MODEL_PATH}"


@pytest.fixture(scope="module")
def session():
    """ONNX Runtime の推論セッションを返す."""
    return ort.InferenceSession(str(MODEL_PATH))


def _run(session, img1: np.ndarray, img2: np.ndarray, alpha: float) -> np.ndarray:
    return session.run(None, {
        "input": img1,
        "input2": img2,
        "alpha": np.array([alpha], dtype=np.float32),
    })[0]


def _numpy_alpha_blend(img1, img2, alpha):
    """NumPy 参照実装."""
    return alpha * img1 + (1 - alpha) * img2


class TestAlphaBlendOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, img, 0.5)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, img, 0.5)
        assert out.shape == (2, 3, 8, 8)


class TestAlphaBlendValues:
    """出力値の正確性を検証するテスト群."""

    def test_alpha_zero(self, session):
        """alpha=0 で input2 がそのまま出力されること."""
        img1 = np.full((1, 3, 4, 4), 0.2, dtype=np.float32)
        img2 = np.full((1, 3, 4, 4), 0.8, dtype=np.float32)
        out = _run(session, img1, img2, 0.0)
        np.testing.assert_allclose(out, 0.8, atol=1e-5)

    def test_alpha_one(self, session):
        """alpha=1 で input がそのまま出力されること."""
        img1 = np.full((1, 3, 4, 4), 0.2, dtype=np.float32)
        img2 = np.full((1, 3, 4, 4), 0.8, dtype=np.float32)
        out = _run(session, img1, img2, 1.0)
        np.testing.assert_allclose(out, 0.2, atol=1e-5)

    def test_alpha_half(self, session):
        """alpha=0.5 で平均値."""
        img1 = np.full((1, 3, 4, 4), 0.2, dtype=np.float32)
        img2 = np.full((1, 3, 4, 4), 0.8, dtype=np.float32)
        out = _run(session, img1, img2, 0.5)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致すること."""
        rng = np.random.default_rng(42)
        img1 = rng.random((2, 3, 16, 16), dtype=np.float32)
        img2 = rng.random((2, 3, 16, 16), dtype=np.float32)
        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            expected = _numpy_alpha_blend(img1, img2, alpha)
            out = _run(session, img1, img2, alpha)
            np.testing.assert_allclose(out, expected, atol=1e-5)
