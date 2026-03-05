"""レベル補正モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "levels.onnx"


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


def _run(
    session,
    img: np.ndarray,
    in_black: float = 0.0,
    in_white: float = 1.0,
    gamma: float = 1.0,
    out_black: float = 0.0,
    out_white: float = 1.0,
) -> np.ndarray:
    """セッションで推論を実行し、最初の出力を返すヘルパー."""
    return session.run(None, {
        "input": img,
        "in_black": np.array([in_black], dtype=np.float32),
        "in_white": np.array([in_white], dtype=np.float32),
        "gamma": np.array([gamma], dtype=np.float32),
        "out_black": np.array([out_black], dtype=np.float32),
        "out_white": np.array([out_white], dtype=np.float32),
    })[0]


def _numpy_levels(
    img: np.ndarray,
    in_black: float = 0.0,
    in_white: float = 1.0,
    gamma: float = 1.0,
    out_black: float = 0.0,
    out_white: float = 1.0,
) -> np.ndarray:
    """NumPy 参照実装."""
    x = np.clip(img, in_black, in_white)
    x = (x - in_black) / max(in_white - in_black, 1e-6)
    x = np.clip(x, 0, 1)
    x = x ** gamma
    x = x * (out_white - out_black) + out_black
    return np.clip(x, 0, 1)


class TestLevelsOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (2, 3, 8, 8)


class TestLevelsValues:
    """出力値の正確性を検証するテスト群."""

    def test_default_identity(self, session):
        """全パラメータがデフォルト値で恒等変換になること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_in_black_raise(self, session):
        """in_black を引き上げると暗部がクランプされること."""
        img = np.array([[[[0.1, 0.3, 0.5, 0.8]]]], dtype=np.float32)
        img = np.broadcast_to(img, (1, 3, 1, 4)).copy()
        out = _run(session, img, in_black=0.3)
        # 0.1, 0.3 は in_black=0.3 にクランプ → 0.0 にマッピング
        # 0.5 → (0.5-0.3)/(1.0-0.3) ≈ 0.2857
        # 0.8 → (0.8-0.3)/(1.0-0.3) ≈ 0.7143
        expected = _numpy_levels(img, in_black=0.3)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_gamma_brighten(self, session):
        """gamma < 1 で中間値が上昇 (明るくなる) こと."""
        img = np.array([[[[0.25]]]], dtype=np.float32)
        img = np.broadcast_to(img, (1, 3, 1, 1)).copy()
        out = _run(session, img, gamma=0.5)
        # 0.25 ^ 0.5 = 0.5
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_gamma_darken(self, session):
        """gamma > 1 で中間値が低下 (暗くなる) こと."""
        img = np.array([[[[0.5]]]], dtype=np.float32)
        img = np.broadcast_to(img, (1, 3, 1, 1)).copy()
        out = _run(session, img, gamma=2.0)
        # 0.5 ^ 2.0 = 0.25
        np.testing.assert_allclose(out, 0.25, atol=1e-5)

    def test_output_range_shrink(self, session):
        """出力レンジを縮小すると出力が [out_black, out_white] に収まること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img, out_black=0.2, out_white=0.8)
        assert out.min() >= 0.2 - 1e-5
        assert out.max() <= 0.8 + 1e-5

    def test_output_range_01(self, session):
        """出力値が [0, 1] 範囲内であること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        for params in [
            {"in_black": 0.2, "gamma": 0.5},
            {"in_white": 0.8, "gamma": 2.0},
            {"out_black": 0.1, "out_white": 0.9},
        ]:
            out = _run(session, img, **params)
            assert out.min() >= -1e-6
            assert out.max() <= 1.0 + 1e-6

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        param_sets = [
            {},  # デフォルト (恒等変換)
            {"in_black": 0.2, "in_white": 0.9},
            {"gamma": 0.5},
            {"gamma": 2.0},
            {"out_black": 0.1, "out_white": 0.8},
            {"in_black": 0.1, "in_white": 0.9, "gamma": 1.5,
             "out_black": 0.05, "out_white": 0.95},
        ]
        for params in param_sets:
            expected = _numpy_levels(img, **params)
            out = _run(session, img, **params)
            np.testing.assert_allclose(out, expected, atol=1e-5)
