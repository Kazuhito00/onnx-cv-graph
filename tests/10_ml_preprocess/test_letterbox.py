"""Letterbox リサイズモデルのテスト."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_DIR = PROJECT_ROOT / "models" / CATEGORY


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    if not (MODEL_DIR / "letterbox.onnx").exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_DIR / "letterbox.onnx"))


def _run(session, img: np.ndarray, target_h: float = 640.0,
         target_w: float = 640.0) -> np.ndarray:
    return session.run(None, {
        "input": img,
        "target_h": np.array([target_h], dtype=np.float32),
        "target_w": np.array([target_w], dtype=np.float32),
    })[0]


class TestLetterboxOutputShape:
    def test_output_matches_target(self, session):
        """出力が target_h × target_w になる."""
        img = np.random.rand(1, 3, 100, 200).astype(np.float32)
        out = _run(session, img, target_h=128, target_w=128)
        assert out.shape == (1, 3, 128, 128)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 100, 100).astype(np.float32)
        out = _run(session, img, target_h=64, target_w=64)
        assert out.shape == (2, 3, 64, 64)

    def test_square_to_square(self, session):
        """正方形入力 → 正方形出力."""
        img = np.random.rand(1, 3, 64, 64).astype(np.float32)
        out = _run(session, img, target_h=128, target_w=128)
        assert out.shape == (1, 3, 128, 128)


class TestLetterboxValues:
    def test_padding_color(self, session):
        """パディング部分が 0.5 (グレー) であること."""
        # 横長画像 → 上下にパディング
        img = np.full((1, 3, 64, 128), 0.0, dtype=np.float32)
        out = _run(session, img, target_h=128, target_w=128)
        # scale = min(128/64, 128/128) = min(2.0, 1.0) = 1.0
        # resized: 64×128 → パディング: (128-64)/2 = 32 上下
        # 上部パディング行が 0.5 であること
        pad_region = out[0, 0, 0:32, :]
        np.testing.assert_allclose(pad_region, 0.5, atol=1e-5)

    def test_image_not_distorted(self, session):
        """正方形入力 → 正方形ターゲットでパディングが最小."""
        img = np.full((1, 3, 64, 64), 0.3, dtype=np.float32)
        out = _run(session, img, target_h=128, target_w=128)
        # scale = min(128/64, 128/64) = 2.0 → resized = 128×128 → パディングなし
        # 全域が画像データ (bilinear 補間で 0.3 に近い)
        np.testing.assert_allclose(out, 0.3, atol=0.01)

    def test_aspect_ratio_preserved_landscape(self, session):
        """横長入力でアスペクト比が維持される."""
        img = np.random.rand(1, 3, 64, 128).astype(np.float32)
        out = _run(session, img, target_h=128, target_w=128)
        # scale = min(128/64, 128/128) = 1.0
        # resized: 64×128, pad_h=(128-64)=64, pad_top=32
        # パディング行が 0.5
        np.testing.assert_allclose(out[0, :, 0:32, :], 0.5, atol=1e-5)
        np.testing.assert_allclose(out[0, :, 96:128, :], 0.5, atol=1e-5)

    def test_aspect_ratio_preserved_portrait(self, session):
        """縦長入力でアスペクト比が維持される."""
        img = np.random.rand(1, 3, 128, 64).astype(np.float32)
        out = _run(session, img, target_h=128, target_w=128)
        # scale = min(128/128, 128/64) = 1.0
        # resized: 128×64, pad_w=(128-64)=64, pad_left=32
        np.testing.assert_allclose(out[0, :, :, 0:32], 0.5, atol=1e-5)
        np.testing.assert_allclose(out[0, :, :, 96:128], 0.5, atol=1e-5)

    def test_output_range(self, session):
        """出力が [0, 1] 範囲内."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 80, 120), dtype=np.float32)
        out = _run(session, img, target_h=128, target_w=128)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6
