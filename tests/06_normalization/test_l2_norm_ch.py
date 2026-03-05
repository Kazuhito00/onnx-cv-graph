"""チャネル軸 L2 正規化モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "l2_norm_ch.onnx"


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


def _numpy_l2_norm_ch(img: np.ndarray) -> np.ndarray:
    """NumPy 参照実装: チャネル軸のみで L2 正規化."""
    l2 = np.sqrt((img ** 2).sum(axis=1, keepdims=True))
    result = img / (l2 + 1e-8)
    return np.clip(result, 0.0, 1.0)


class TestL2NormChOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (2, 3, 8, 8)


class TestL2NormChValues:
    """出力値の正確性を検証するテスト群."""

    def test_output_range(self, session):
        """出力が [0, 1] 範囲内であること."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_per_pixel_l2_norm(self, session):
        """各ピクセルの L2 ノルムが 1 に近いこと."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        # 各ピクセルの L2 ノルム (axis=1)
        l2_per_pixel = np.sqrt((out ** 2).sum(axis=1))
        np.testing.assert_allclose(l2_per_pixel, 1.0, atol=1e-4)

    def test_uniform_image(self, session):
        """均一画像で各チャネルが 1/sqrt(3) になること."""
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, img)
        expected_val = 1.0 / np.sqrt(3.0)
        np.testing.assert_allclose(out, expected_val, atol=1e-4)

    def test_all_black(self, session):
        """全黒画像で出力が 0 になること."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        expected = _numpy_l2_norm_ch(img)
        out = _run(session, img)
        np.testing.assert_allclose(out, expected, atol=1e-5)
