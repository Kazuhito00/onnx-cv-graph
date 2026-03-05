"""ガンマ補正モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "gamma.onnx"


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


def _run(session, img: np.ndarray, gamma: float) -> np.ndarray:
    """セッションで推論を実行し、最初の出力を返すヘルパー."""
    g = np.array([gamma], dtype=np.float32)
    return session.run(None, {"input": img, "gamma": g})[0]


def _numpy_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """NumPy 参照実装: input ^ gamma をクリップ."""
    return np.clip(np.power(img, gamma), 0.0, 1.0)


class TestGammaOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 2.2)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.5)
        assert out.shape == (2, 3, 8, 8)


class TestGammaValues:
    """出力値の正確性を検証するテスト群."""

    def test_gamma_one_unchanged(self, session):
        """gamma=1.0 で入力と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 1.0)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_gamma_less_than_one_brightens(self, session):
        """gamma < 1 で中間値が明るくなること."""
        img = np.full((1, 3, 4, 4), 0.25, dtype=np.float32)
        out = _run(session, img, 0.5)
        # 0.25 ^ 0.5 = 0.5
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_gamma_greater_than_one_darkens(self, session):
        """gamma > 1 で中間値が暗くなること."""
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, img, 2.0)
        # 0.5 ^ 2 = 0.25
        np.testing.assert_allclose(out, 0.25, atol=1e-5)

    def test_zero_stays_zero(self, session):
        """0.0 は任意の gamma で 0.0 のままであること."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img, 2.2)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_one_stays_one(self, session):
        """1.0 は任意の gamma で 1.0 のままであること."""
        img = np.ones((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img, 0.5)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_output_range(self, session):
        """出力値が [0, 1] 範囲内であること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        for g in [0.1, 0.5, 1.0, 2.2, 5.0]:
            out = _run(session, img, g)
            assert out.min() >= -1e-6
            assert out.max() <= 1.0 + 1e-6

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        for g in [0.5, 1.0, 2.2]:
            expected = _numpy_gamma(img, g)
            out = _run(session, img, g)
            np.testing.assert_allclose(out, expected, atol=1e-5)
