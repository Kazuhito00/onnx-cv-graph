"""Gray World ホワイトバランスモデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "wb_gray_world.onnx"


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


def _numpy_gray_world(img):
    """NumPy 参照実装."""
    eps = 1e-7
    ch_mean = img.mean(axis=(2, 3), keepdims=True)  # (N, 3, 1, 1)
    global_mean = ch_mean.mean(axis=1, keepdims=True)  # (N, 1, 1, 1)
    gain = global_mean / (ch_mean + eps)
    return np.clip(img * gain, 0.0, 1.0).astype(np.float32)


class TestWbGrayWorldOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        assert _run(session, img).shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        assert _run(session, img).shape == (2, 3, 8, 8)


class TestWbGrayWorldValues:
    def test_gray_image_unchanged(self, session):
        """全チャネル同値の画像は変化なし."""
        img = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_channel_means_equalized(self, session):
        """補正後の各チャネル平均が近くなること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 32, 32), dtype=np.float32)
        # R チャネルを明るく偏らせる
        img[:, 0] = np.clip(img[:, 0] + 0.3, 0, 1)
        out = _run(session, img)
        means = out.mean(axis=(2, 3))  # (1, 3)
        # 3チャネルの平均のばらつきが小さくなること
        assert means.std() < img.mean(axis=(2, 3)).std()

    def test_matches_numpy_reference(self, session):
        """NumPy 参照実装と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        expected = _numpy_gray_world(img)
        out = _run(session, img)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_output_range(self, session):
        """出力値が [0, 1] 範囲内."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_already_balanced(self, session):
        """既にバランスの取れた画像は大きく変わらないこと."""
        rng = np.random.default_rng(42)
        # 全チャネル同じ分布
        ch = rng.random((1, 1, 16, 16), dtype=np.float32)
        img = np.broadcast_to(ch, (1, 3, 16, 16)).copy()
        out = _run(session, img)
        np.testing.assert_allclose(out, img, atol=1e-4)
