"""任意サイズリサイズモデルのテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_DIR = PROJECT_ROOT / "models" / CATEGORY


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    if not (MODEL_DIR / "resize_to.onnx").exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_DIR / "resize_to.onnx"))


def _run(session, img: np.ndarray, target_h: float, target_w: float) -> np.ndarray:
    return session.run(None, {
        "input": img,
        "target_h": np.array([target_h], dtype=np.float32),
        "target_w": np.array([target_w], dtype=np.float32),
    })[0]


class TestResizeToOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_same_size(self, session):
        """同じサイズを指定した場合、形状が変わらないこと."""
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        out = _run(session, img, 16.0, 16.0)
        assert out.shape == (1, 3, 16, 16)

    def test_upscale(self, session):
        """拡大リサイズで正しい出力サイズになること."""
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 32.0, 24.0)
        assert out.shape == (1, 3, 32, 24)

    def test_downscale(self, session):
        """縮小リサイズで正しい出力サイズになること."""
        img = np.random.rand(1, 3, 32, 32).astype(np.float32)
        out = _run(session, img, 8.0, 16.0)
        assert out.shape == (1, 3, 8, 16)

    def test_non_square(self, session):
        """非正方形の入出力で正しい形状になること."""
        img = np.random.rand(1, 3, 12, 20).astype(np.float32)
        out = _run(session, img, 24.0, 10.0)
        assert out.shape == (1, 3, 24, 10)

    def test_batch(self, session):
        """バッチ入力で正しい形状になること."""
        img = np.random.rand(3, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 16.0, 16.0)
        assert out.shape == (3, 3, 16, 16)


class TestResizeToValues:
    """出力値の正確性を検証するテスト群."""

    def test_same_size_identity(self, session):
        """同じサイズ指定で入力と一致すること."""
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        out = _run(session, img, 16.0, 16.0)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_uniform_image(self, session):
        """均一画像はリサイズ後も同一値であること."""
        img = np.full((1, 3, 8, 8), 0.7, dtype=np.float32)
        out = _run(session, img, 32.0, 24.0)
        np.testing.assert_allclose(out, 0.7, atol=1e-5)

    def test_output_range(self, session):
        """出力値が [0, 1] 範囲内であること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img, 32.0, 8.0)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6


class TestResizeToVsOpenCV:
    """OpenCV resize との比較テスト群."""

    def test_matches_opencv_upscale(self, session):
        """拡大で OpenCV bilinear resize との一致を検証."""
        rng = np.random.default_rng(42)
        img_nchw = rng.random((1, 3, 16, 16), dtype=np.float32)

        onnx_out = _run(session, img_nchw, 32.0, 24.0)
        onnx_uint8 = (onnx_out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)

        hwc_u8 = (img_nchw[0].transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)
        cv_out = cv2.resize(hwc_u8, (24, 32), interpolation=cv2.INTER_LINEAR)

        diff = np.abs(onnx_uint8.astype(np.int16) - cv_out.astype(np.int16))
        assert diff.max() <= 2, f"最大誤差 {diff.max()} > 2"

    def test_matches_opencv_downscale(self, session):
        """縮小で OpenCV bilinear resize との一致を検証."""
        rng = np.random.default_rng(123)
        img_nchw = rng.random((1, 3, 32, 32), dtype=np.float32)

        onnx_out = _run(session, img_nchw, 8.0, 16.0)
        onnx_uint8 = (onnx_out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)

        hwc_u8 = (img_nchw[0].transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)
        cv_out = cv2.resize(hwc_u8, (16, 8), interpolation=cv2.INTER_LINEAR)

        diff = np.abs(onnx_uint8.astype(np.int16) - cv_out.astype(np.int16))
        assert diff.max() <= 2, f"最大誤差 {diff.max()} > 2"
