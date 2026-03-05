"""加重加算モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "weighted_add.onnx"


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


def _run(session, img1: np.ndarray, img2: np.ndarray,
         alpha: float, beta: float, gamma: float) -> np.ndarray:
    return session.run(None, {
        "input": img1,
        "input2": img2,
        "alpha": np.array([alpha], dtype=np.float32),
        "beta": np.array([beta], dtype=np.float32),
        "gamma": np.array([gamma], dtype=np.float32),
    })[0]


def _numpy_weighted_add(img1, img2, alpha, beta, gamma):
    """NumPy 参照実装."""
    return np.clip(alpha * img1 + beta * img2 + gamma, 0.0, 1.0)


class TestWeightedAddOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, img, 1.0, 1.0, 0.0)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, img, 1.0, 1.0, 0.0)
        assert out.shape == (2, 3, 8, 8)


class TestWeightedAddValues:
    """出力値の正確性を検証するテスト群."""

    def test_output_range(self, session):
        """出力が [0, 1] 範囲内であること."""
        rng = np.random.default_rng(42)
        img1 = rng.random((1, 3, 8, 8), dtype=np.float32)
        img2 = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img1, img2, 1.5, 1.5, -0.5)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_alpha_only(self, session):
        """beta=0, gamma=0 で alpha*input のみ."""
        rng = np.random.default_rng(42)
        img1 = rng.random((1, 3, 8, 8), dtype=np.float32)
        img2 = np.zeros_like(img1)
        out = _run(session, img1, img2, 0.5, 0.0, 0.0)
        expected = np.clip(0.5 * img1, 0.0, 1.0)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_equal_blend(self, session):
        """alpha=0.5, beta=0.5 で平均."""
        img1 = np.full((1, 3, 4, 4), 0.2, dtype=np.float32)
        img2 = np.full((1, 3, 4, 4), 0.8, dtype=np.float32)
        out = _run(session, img1, img2, 0.5, 0.5, 0.0)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_gamma_offset(self, session):
        """gamma で出力がオフセットされること."""
        img = np.full((1, 3, 4, 4), 0.3, dtype=np.float32)
        out = _run(session, img, img, 1.0, 0.0, 0.2)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致すること."""
        rng = np.random.default_rng(42)
        img1 = rng.random((2, 3, 16, 16), dtype=np.float32)
        img2 = rng.random((2, 3, 16, 16), dtype=np.float32)
        expected = _numpy_weighted_add(img1, img2, 0.7, 0.3, 0.1)
        out = _run(session, img1, img2, 0.7, 0.3, 0.1)
        np.testing.assert_allclose(out, expected, atol=1e-5)
