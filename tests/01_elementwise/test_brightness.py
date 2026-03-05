"""明るさ調整モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "brightness.onnx"


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


def _run(session, img: np.ndarray, brightness: float) -> np.ndarray:
    """セッションで推論を実行し、最初の出力を返すヘルパー."""
    b = np.array([brightness], dtype=np.float32)
    return session.run(None, {"input": img, "brightness": b})[0]


def _numpy_brightness(img: np.ndarray, brightness: float) -> np.ndarray:
    """NumPy 参照実装: input + brightness をクリップ."""
    return np.clip(img + brightness, 0.0, 1.0)


class TestBrightnessOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.1)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.1)
        assert out.shape == (2, 3, 8, 8)


class TestBrightnessValues:
    """出力値の正確性を検証するテスト群."""

    def test_zero_brightness_unchanged(self, session):
        """brightness=0 で入力と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 0.0)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_positive_brightness(self, session):
        """正の brightness で全画素が明るくなること."""
        img = np.full((1, 3, 4, 4), 0.3, dtype=np.float32)
        out = _run(session, img, 0.2)
        np.testing.assert_allclose(out, 0.5, atol=1e-6)

    def test_negative_brightness(self, session):
        """負の brightness で全画素が暗くなること."""
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, img, -0.3)
        np.testing.assert_allclose(out, 0.2, atol=1e-6)

    def test_clip_upper(self, session):
        """上限 1.0 でクリップされること."""
        img = np.full((1, 3, 4, 4), 0.8, dtype=np.float32)
        out = _run(session, img, 0.5)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_clip_lower(self, session):
        """下限 0.0 でクリップされること."""
        img = np.full((1, 3, 4, 4), 0.2, dtype=np.float32)
        out = _run(session, img, -0.5)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_output_range(self, session):
        """出力値が [0, 1] 範囲内であること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        for b in [-0.5, 0.0, 0.5]:
            out = _run(session, img, b)
            assert out.min() >= -1e-6
            assert out.max() <= 1.0 + 1e-6

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        for b in [-0.3, 0.0, 0.4]:
            expected = _numpy_brightness(img, b)
            out = _run(session, img, b)
            np.testing.assert_allclose(out, expected, atol=1e-5)
