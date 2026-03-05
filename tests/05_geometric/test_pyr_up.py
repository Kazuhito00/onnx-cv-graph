"""ピラミッドアップモデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "pyr_up.onnx"


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


def _run(session, img):
    return session.run(None, {"input": img})[0]


class TestPyrUpOutputShape:
    def test_double_size(self, session):
        """出力が入力の2倍のサイズ."""
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 32, 32)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (2, 3, 16, 16)


class TestPyrUpValues:
    def test_uniform_unchanged(self, session):
        """均一画像は拡大後も同じ値."""
        img = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.5, atol=1e-2)

    def test_output_range(self, session):
        """出力値が [0, 1] 範囲内."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_smoothing_effect(self, session):
        """ぼかし効果: 拡大画像は元画像より滑らか."""
        img = np.zeros((1, 3, 8, 8), dtype=np.float32)
        img[:, :, ::2, ::2] = 1.0  # チェッカーボード
        out = _run(session, img)
        # 隣接ピクセル差の平均が小さい (滑らか)
        h_diff = np.abs(np.diff(out, axis=3)).mean()
        assert h_diff < 0.5

    def test_down_up_approximate_identity(self, session):
        """pyrDown → pyrUp で元画像に近い (完全一致ではない)."""
        down_path = PROJECT_ROOT / "models" / CATEGORY / "pyr_down.onnx"
        if not down_path.exists():
            pytest.skip("pyr_down モデルが見つかりません")
        down_sess = ort.InferenceSession(str(down_path))

        img = np.full((1, 3, 32, 32), 0.5, dtype=np.float32)
        img[:, :, 10:22, 10:22] = 0.8  # 中央に明るい矩形

        down = down_sess.run(None, {"input": img})[0]
        up = _run(session, down)

        # サイズが元画像と同じ
        assert up.shape == img.shape
        # 大まかに元画像に近い (エッジ部分はぼけるため許容誤差大)
        np.testing.assert_allclose(up, img, atol=0.2)
