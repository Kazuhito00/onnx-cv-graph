"""セピア調変換モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "sepia.onnx"

# Microsoft 標準セピア変換行列
SEPIA_MATRIX = np.array([
    [0.393, 0.769, 0.189],  # R'
    [0.349, 0.686, 0.168],  # G'
    [0.272, 0.534, 0.131],  # B'
], dtype=np.float32)


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
    """セッションで推論を実行し、最初の出力を返すヘルパー."""
    return session.run(None, {"input": img})[0]


def _numpy_sepia(img: np.ndarray) -> np.ndarray:
    """NumPy 参照実装: セピア変換行列を適用してクリップ."""
    n, c, h, w = img.shape
    # (N, 3, H*W) に reshape → 行列積 → 元に戻す
    flat = img.reshape(n, 3, h * w)
    result = SEPIA_MATRIX @ flat  # (N, 3, H*W)
    return np.clip(result.reshape(n, 3, h, w), 0.0, 1.0)


class TestSepiaOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (2, 3, 8, 8)


class TestSepiaValues:
    """出力値の正確性を検証するテスト群."""

    def test_all_black(self, session):
        """全黒画像はセピア変換後も全黒であること."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_array_equal(out, 0.0)

    def test_all_white_clipped(self, session):
        """全白画像ではクリップにより 1.0 を超えないこと."""
        img = np.ones((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img)
        # R' = 0.393+0.769+0.189 = 1.351 → clip → 1.0
        # G' = 0.349+0.686+0.168 = 1.203 → clip → 1.0
        # B' = 0.272+0.534+0.131 = 0.937
        np.testing.assert_allclose(out[0, 0], 1.0, atol=1e-5)  # R clipped
        np.testing.assert_allclose(out[0, 1], 1.0, atol=1e-5)  # G clipped
        np.testing.assert_allclose(out[0, 2], 0.937, atol=1e-3)  # B

    def test_pure_red(self, session):
        """純赤 (1,0,0) のセピア変換結果."""
        img = np.zeros((1, 3, 1, 1), dtype=np.float32)
        img[0, 0] = 1.0
        out = _run(session, img)
        np.testing.assert_allclose(out[0, 0, 0, 0], 0.393, atol=1e-3)
        np.testing.assert_allclose(out[0, 1, 0, 0], 0.349, atol=1e-3)
        np.testing.assert_allclose(out[0, 2, 0, 0], 0.272, atol=1e-3)

    def test_output_range(self, session):
        """出力値が [0, 1] 範囲内であること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_warm_tone(self, session):
        """セピア変換で R >= G >= B の暖色階調になること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        # 全画素で R' >= G' >= B' (セピア行列の性質)
        assert (out[0, 0] >= out[0, 1] - 1e-6).all(), "R >= G でない画素がある"
        assert (out[0, 1] >= out[0, 2] - 1e-6).all(), "G >= B でない画素がある"

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        expected = _numpy_sepia(img)
        out = _run(session, img)
        np.testing.assert_allclose(out, expected, atol=1e-5)
