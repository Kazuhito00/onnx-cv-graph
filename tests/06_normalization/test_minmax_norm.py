"""Min-Max 正規化モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "minmax_norm.onnx"


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


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


def _numpy_minmax(img: np.ndarray) -> np.ndarray:
    """NumPy 参照実装."""
    # 各画像 (N 軸) ごとに C,H,W で min/max を取る
    x_min = img.min(axis=(1, 2, 3), keepdims=True)
    x_max = img.max(axis=(1, 2, 3), keepdims=True)
    x_range = x_max - x_min
    x_range = np.where(x_range == 0, 1e-8, x_range)
    return (img - x_min) / x_range


class TestMinMaxNormOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (2, 3, 8, 8)


class TestMinMaxNormValues:
    """出力値の正確性を検証するテスト群."""

    def test_output_range(self, session):
        """出力が [0, 1] 範囲内であること."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_output_min_zero_max_one(self, session):
        """出力の最小値が 0、最大値が 1 に近いこと."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out.min(), 0.0, atol=1e-5)
        np.testing.assert_allclose(out.max(), 1.0, atol=1e-5)

    def test_uniform_image(self, session):
        """均一画像では全値がほぼ 0 になること (range ≈ 0)."""
        img = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-2)

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        expected = _numpy_minmax(img)
        out = _run(session, img)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_already_normalized(self, session):
        """[0, 1] の全範囲を含む画像では入力と近い値になること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        img[0, 0, 0, 0] = 0.0
        img[0, 0, 0, 1] = 1.0
        out = _run(session, img)
        expected = _numpy_minmax(img)
        np.testing.assert_allclose(out, expected, atol=1e-5)
