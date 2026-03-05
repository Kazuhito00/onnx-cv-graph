"""White Patch ホワイトバランスモデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "wb_white_patch.onnx"


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


def _numpy_white_patch(img):
    """NumPy 参照実装."""
    eps = 1e-7
    ch_max = img.max(axis=(2, 3), keepdims=True)  # (N, 3, 1, 1)
    gain = 1.0 / (ch_max + eps)
    return np.clip(img * gain, 0.0, 1.0).astype(np.float32)


class TestWbWhitePatchOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        assert _run(session, img).shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        assert _run(session, img).shape == (2, 3, 8, 8)


class TestWbWhitePatchValues:
    def test_already_white_unchanged(self, session):
        """各チャネルの最大値が既に 1.0 なら変化なし."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        # 各チャネルに 1.0 のピクセルを設定
        img[0, 0, 0, 0] = 1.0
        img[0, 1, 0, 0] = 1.0
        img[0, 2, 0, 0] = 1.0
        out = _run(session, img)
        np.testing.assert_allclose(out, img, atol=1e-4)

    def test_channel_max_becomes_one(self, session):
        """補正後の各チャネル最大値がほぼ 1.0 になること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32) * 0.5
        # 最大値を各チャネルで異なる値に
        img[0, 0, 0, 0] = 0.8
        img[0, 1, 0, 0] = 0.5
        img[0, 2, 0, 0] = 0.3
        out = _run(session, img)
        for ch in range(3):
            np.testing.assert_allclose(out[0, ch].max(), 1.0, atol=1e-4)

    def test_matches_numpy_reference(self, session):
        """NumPy 参照実装と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        expected = _numpy_white_patch(img)
        out = _run(session, img)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_output_range(self, session):
        """出力値が [0, 1] 範囲内."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_uniform_scales_to_one(self, session):
        """均一値 0.5 → 全画素 1.0."""
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 1.0, atol=1e-4)
