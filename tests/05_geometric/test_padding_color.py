"""任意色パディング (padding_color) モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "padding_color.onnx"


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    if not MODEL_PATH.exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_PATH))


def _run(session, img, pad_ratio=0.1, pad_r=0.0, pad_g=0.0, pad_b=0.0):
    return session.run(None, {
        "input": img,
        "pad_ratio": np.array([pad_ratio], dtype=np.float32),
        "pad_r": np.array([pad_r], dtype=np.float32),
        "pad_g": np.array([pad_g], dtype=np.float32),
        "pad_b": np.array([pad_b], dtype=np.float32),
    })[0]


class TestPaddingColorOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_ratio_zero_same_size(self, session):
        """pad_ratio=0 で入力と同じサイズ."""
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        out = _run(session, img, 0.0)
        assert out.shape == (1, 3, 16, 16)

    def test_ratio_adds_pixels(self, session):
        """pad_ratio=0.25 で H,W が 25% ずつ上下左右に追加."""
        img = np.random.rand(1, 3, 16, 20).astype(np.float32)
        out = _run(session, img, 0.25)
        # pad_h = floor(16*0.25)=4, pad_w = floor(20*0.25)=5
        assert out.shape == (1, 3, 16 + 4 * 2, 20 + 5 * 2)

    def test_batch(self, session):
        """バッチ処理で正しい形状."""
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.25)
        assert out.shape[0] == 2
        assert out.shape[1] == 3


class TestPaddingColorValues:
    """出力値の正確性を検証するテスト群."""

    def test_center_preserved(self, session):
        """中央部分は元画像と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 20), dtype=np.float32)
        out = _run(session, img, 0.25, pad_r=0.5, pad_g=0.3, pad_b=0.1)
        pad_h, pad_w = 4, 5
        center = out[:, :, pad_h:-pad_h, pad_w:-pad_w]
        np.testing.assert_allclose(center, img, atol=1e-6)

    def test_black_padding(self, session):
        """pad_r=pad_g=pad_b=0 でパディング部分が黒."""
        img = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)
        out = _run(session, img, 0.25, pad_r=0.0, pad_g=0.0, pad_b=0.0)
        pad_h = 2  # floor(8*0.25)
        # パディング上端行は黒
        np.testing.assert_allclose(out[:, :, 0, :], 0.0, atol=1e-6)
        # 中央は元画像の値
        np.testing.assert_allclose(out[:, :, pad_h, pad_h], 0.5, atol=1e-6)

    def test_white_padding(self, session):
        """pad_r=pad_g=pad_b=1 でパディング部分が白."""
        rng = np.random.default_rng(10)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 0.25, pad_r=1.0, pad_g=1.0, pad_b=1.0)
        pad_h = 2
        # パディング上端行は白
        np.testing.assert_allclose(out[:, :, 0, :], 1.0, atol=1e-6)

    def test_red_padding(self, session):
        """pad_r=1, pad_g=0, pad_b=0 でパディング部分が赤."""
        img = np.zeros((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 0.25, pad_r=1.0, pad_g=0.0, pad_b=0.0)
        pad_h = 2
        # パディング上端行: R=1, G=0, B=0
        np.testing.assert_allclose(out[0, 0, 0, :], 1.0, atol=1e-6)  # R
        np.testing.assert_allclose(out[0, 1, 0, :], 0.0, atol=1e-6)  # G
        np.testing.assert_allclose(out[0, 2, 0, :], 0.0, atol=1e-6)  # B

    def test_custom_color_padding(self, session):
        """任意色 (0.2, 0.4, 0.6) でパディング."""
        img = np.zeros((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 0.25, pad_r=0.2, pad_g=0.4, pad_b=0.6)
        pad_h = 2
        # パディング上端行の各チャネル
        np.testing.assert_allclose(out[0, 0, 0, :], 0.2, atol=1e-6)  # R
        np.testing.assert_allclose(out[0, 1, 0, :], 0.4, atol=1e-6)  # G
        np.testing.assert_allclose(out[0, 2, 0, :], 0.6, atol=1e-6)  # B

    def test_ratio_zero_identity(self, session):
        """pad_ratio=0 で入力と完全一致."""
        rng = np.random.default_rng(99)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img, 0.0, pad_r=1.0, pad_g=0.0, pad_b=0.0)
        np.testing.assert_allclose(out, img, atol=1e-6)


class TestPaddingColorVsOpenCV:
    """OpenCV copyMakeBorder との比較テスト群."""

    def test_matches_opencv(self, session):
        """OpenCV copyMakeBorder(BORDER_CONSTANT) との一致を検証."""
        H, W = 16, 20
        pad_ratio = 0.25
        pad_h = int(np.floor(H * pad_ratio))
        pad_w = int(np.floor(W * pad_ratio))
        r, g, b = 0.3, 0.5, 0.7

        rng = np.random.default_rng(42)
        img_nchw = rng.random((1, 3, H, W), dtype=np.float32)

        onnx_out = _run(session, img_nchw, pad_ratio, pad_r=r, pad_g=g, pad_b=b)
        onnx_u8 = (onnx_out[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)

        hwc_u8 = (img_nchw[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        # OpenCV は BGR 順
        cv_out = cv2.copyMakeBorder(
            hwc_u8, pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT,
            value=(round(r * 255), round(g * 255), round(b * 255)),
        )

        diff = np.abs(onnx_u8.astype(np.int16) - cv_out.astype(np.int16))
        assert diff.max() <= 1, f"最大誤差 {diff.max()} > 1"
