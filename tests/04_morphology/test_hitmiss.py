"""ヒットオアミス変換モデルのテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
十字型構造要素 (3×3) を使用.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "hitmiss_3x3.onnx"


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    if not MODEL_PATH.exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )
    assert MODEL_PATH.exists()


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_PATH))


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


def _make_binary_3ch(gray_2d: np.ndarray) -> np.ndarray:
    """2D バイナリ配列 → (1, 3, H, W) float32."""
    return np.broadcast_to(
        gray_2d.astype(np.float32)[np.newaxis, np.newaxis, :, :],
        (1, 3, gray_2d.shape[0], gray_2d.shape[1]),
    ).copy()


class TestHitMissOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (2, 3, 8, 8)


class TestHitMissValues:
    def test_all_black_zero(self, session):
        """全黒画像 → 出力ゼロ."""
        img = np.zeros((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_array_equal(out, 0.0)

    def test_all_white_zero(self, session):
        """全白画像 → 出力ゼロ (背景条件が満たされない)."""
        img = np.ones((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_array_equal(out, 0.0)

    def test_cross_pattern_detected(self, session):
        """十字型パターンが検出されること.

        十字型構造要素:
          fg: [[0,1,0],[1,1,1],[0,1,0]]
          bg: [[1,0,1],[0,0,0],[1,0,1]]
        → 十字が白(1)で角が黒(0)のパターンにマッチ.
        """
        # 8x8 の全黒背景に十字型を配置
        gray = np.zeros((8, 8), dtype=np.float32)
        # (3,4) を中心に十字を描画
        r, c = 3, 4
        gray[r, c] = 1.0
        gray[r - 1, c] = 1.0
        gray[r + 1, c] = 1.0
        gray[r, c - 1] = 1.0
        gray[r, c + 1] = 1.0
        # 角は 0 のまま

        img = _make_binary_3ch(gray)
        out = _run(session, img)
        # 中央ピクセルが検出される
        assert out[0, 0, r, c] == 1.0

    def test_output_binary(self, session):
        """出力が 0.0 or 1.0 のみであること."""
        rng = np.random.default_rng(42)
        # ランダムな二値画像
        img = (rng.random((1, 3, 16, 16)) > 0.5).astype(np.float32)
        out = _run(session, img)
        unique = np.unique(out)
        for v in unique:
            assert v in [0.0, 1.0], f"非バイナリ値: {v}"

    def test_3ch_identical(self, session):
        """全3チャネルが同一値であること."""
        rng = np.random.default_rng(42)
        img = (rng.random((1, 3, 16, 16)) > 0.5).astype(np.float32)
        out = _run(session, img)
        np.testing.assert_array_equal(out[0, 0], out[0, 1])
        np.testing.assert_array_equal(out[0, 0], out[0, 2])
