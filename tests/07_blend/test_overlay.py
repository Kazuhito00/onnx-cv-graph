"""オーバーレイ合成モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "overlay.onnx"


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


def _run(session, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    return session.run(None, {
        "input": img1,
        "input2": img2,
    })[0]


def _numpy_overlay(base, blend):
    """NumPy 参照実装."""
    dark = 2 * base * blend
    light = 1 - 2 * (1 - base) * (1 - blend)
    result = np.where(base < 0.5, dark, light)
    return np.clip(result, 0.0, 1.0)


class TestOverlayOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, img)
        assert out.shape == (2, 3, 8, 8)


class TestOverlayValues:
    """出力値の正確性を検証するテスト群."""

    def test_output_range(self, session):
        """出力が [0, 1] 範囲内であること."""
        rng = np.random.default_rng(42)
        img1 = rng.random((2, 3, 16, 16), dtype=np.float32)
        img2 = rng.random((2, 3, 16, 16), dtype=np.float32)
        out = _run(session, img1, img2)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_dark_region(self, session):
        """base < 0.5 で 2*base*blend の計算になること."""
        base = np.full((1, 3, 4, 4), 0.2, dtype=np.float32)
        blend = np.full((1, 3, 4, 4), 0.3, dtype=np.float32)
        out = _run(session, base, blend)
        expected = 2 * 0.2 * 0.3  # = 0.12
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_light_region(self, session):
        """base >= 0.5 で 1 - 2*(1-base)*(1-blend) の計算になること."""
        base = np.full((1, 3, 4, 4), 0.7, dtype=np.float32)
        blend = np.full((1, 3, 4, 4), 0.8, dtype=np.float32)
        out = _run(session, base, blend)
        expected = 1 - 2 * (1 - 0.7) * (1 - 0.8)  # = 0.88
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_neutral_blend(self, session):
        """blend=0.5 でオーバーレイは元画像に近い出力."""
        base = np.full((1, 3, 4, 4), 0.3, dtype=np.float32)
        blend = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, base, blend)
        # base < 0.5 → 2 * 0.3 * 0.5 = 0.3
        np.testing.assert_allclose(out, 0.3, atol=1e-5)

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致すること."""
        rng = np.random.default_rng(42)
        img1 = rng.random((2, 3, 16, 16), dtype=np.float32)
        img2 = rng.random((2, 3, 16, 16), dtype=np.float32)
        expected = _numpy_overlay(img1, img2)
        out = _run(session, img1, img2)
        np.testing.assert_allclose(out, expected, atol=1e-5)
