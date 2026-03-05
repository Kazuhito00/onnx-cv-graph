"""コントラスト調整モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "contrast.onnx"


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


def _run(session, img: np.ndarray, contrast: float, center: float) -> np.ndarray:
    """セッションで推論を実行し、最初の出力を返すヘルパー."""
    c = np.array([contrast], dtype=np.float32)
    ctr = np.array([center], dtype=np.float32)
    return session.run(None, {"input": img, "contrast": c, "center": ctr})[0]


def _numpy_contrast(img: np.ndarray, contrast: float, center: float) -> np.ndarray:
    """NumPy 参照実装: (input - center) * contrast + center をクリップ."""
    return np.clip((img - center) * contrast + center, 0.0, 1.0)


class TestContrastOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 1.5, 0.5)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 1.5, 0.5)
        assert out.shape == (2, 3, 8, 8)


class TestContrastValues:
    """出力値の正確性を検証するテスト群."""

    def test_contrast_one_unchanged(self, session):
        """contrast=1.0 で入力と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 1.0, 0.5)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_contrast_zero_flat(self, session):
        """contrast=0 で全画素が center になること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 0.0, 0.5)
        np.testing.assert_allclose(out, 0.5, atol=1e-6)

    def test_contrast_high(self, session):
        """contrast > 1 で center から離れた値がより離れること."""
        img = np.array([[[[0.3]], [[0.5]], [[0.7]]]], dtype=np.float32)
        out = _run(session, img, 2.0, 0.5)
        # (0.3 - 0.5) * 2 + 0.5 = 0.1
        # (0.5 - 0.5) * 2 + 0.5 = 0.5
        # (0.7 - 0.5) * 2 + 0.5 = 0.9
        expected = np.array([[[[0.1]], [[0.5]], [[0.9]]]], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_output_range(self, session):
        """出力値が [0, 1] 範囲内であること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img, 3.0, 0.5)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        for c, ctr in [(0.5, 0.5), (1.0, 0.5), (2.0, 0.3)]:
            expected = _numpy_contrast(img, c, ctr)
            out = _run(session, img, c, ctr)
            np.testing.assert_allclose(out, expected, atol=1e-5)
