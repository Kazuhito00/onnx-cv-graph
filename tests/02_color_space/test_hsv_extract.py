"""HSV チャネル抽出モデルのテスト.

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

HSV_CHANNELS = [("h", 0), ("s", 1), ("v", 2)]


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    for ch, _ in HSV_CHANNELS:
        p = PROJECT_ROOT / "models" / CATEGORY / f"hsv_{ch}.onnx"
        if not p.exists():
            subprocess.check_call(
                [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
                cwd=str(PROJECT_ROOT),
            )
            break


@pytest.fixture(scope="module", params=HSV_CHANNELS, ids=[ch for ch, _ in HSV_CHANNELS])
def session_and_idx(request):
    ch, idx = request.param
    p = PROJECT_ROOT / "models" / CATEGORY / f"hsv_{ch}.onnx"
    return ort.InferenceSession(str(p)), ch, idx


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


def _opencv_hsv(img_nchw: np.ndarray) -> np.ndarray:
    """OpenCV で HSV に変換し (N, 3, H, W) [0,1] で返す."""
    N, C, H, W = img_nchw.shape
    result = np.zeros_like(img_nchw)
    for i in range(N):
        rgb_hwc = (img_nchw[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        bgr_hwc = cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)
        hsv_hwc = cv2.cvtColor(bgr_hwc, cv2.COLOR_BGR2HSV).astype(np.float32)
        # H: 0-180 → 0-1, S: 0-255 → 0-1, V: 0-255 → 0-1
        hsv_hwc[:, :, 0] /= 180.0
        hsv_hwc[:, :, 1] /= 255.0
        hsv_hwc[:, :, 2] /= 255.0
        result[i] = hsv_hwc.transpose(2, 0, 1)
    return result


class TestHsvExtractOutputShape:
    def test_single_image(self, session_and_idx):
        sess, ch, idx = session_and_idx
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (1, 3, 16, 16)

    def test_batch(self, session_and_idx):
        sess, ch, idx = session_and_idx
        img = np.random.rand(2, 3, 16, 16).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (2, 3, 16, 16)


class TestHsvExtractValues:
    def test_output_range(self, session_and_idx):
        """出力は [0, 1] の範囲内."""
        sess, ch, idx = session_and_idx
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        out = _run(sess, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_all_channels_equal(self, session_and_idx):
        """3チャネル全て同一値."""
        sess, ch, idx = session_and_idx
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        np.testing.assert_array_equal(out[:, 0], out[:, 2])

    def test_v_matches_opencv(self, session_and_idx):
        """V チャネルは OpenCV と近い値になること."""
        sess, ch, idx = session_and_idx
        if ch != "v":
            pytest.skip("V チャネルのみ")
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(sess, img)
        cv_hsv = _opencv_hsv(img)
        # V = max(R,G,B) なので uint8 量子化誤差のみ
        np.testing.assert_allclose(out[:, 0], cv_hsv[:, 2], atol=0.01)

    def test_s_matches_opencv(self, session_and_idx):
        """S チャネルは OpenCV と近い値になること."""
        sess, ch, idx = session_and_idx
        if ch != "s":
            pytest.skip("S チャネルのみ")
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(sess, img)
        cv_hsv = _opencv_hsv(img)
        np.testing.assert_allclose(out[:, 0], cv_hsv[:, 1], atol=0.02)

    def test_pure_red_hue(self):
        """純赤 (1,0,0) の H ≈ 0."""
        p = PROJECT_ROOT / "models" / CATEGORY / "hsv_h.onnx"
        sess = ort.InferenceSession(str(p))
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        img[:, 0] = 1.0  # R=1
        out = _run(sess, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_pure_green_hue(self):
        """純緑 (0,1,0) の H ≈ 120/360 ≈ 0.333."""
        p = PROJECT_ROOT / "models" / CATEGORY / "hsv_h.onnx"
        sess = ort.InferenceSession(str(p))
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        img[:, 1] = 1.0  # G=1
        out = _run(sess, img)
        np.testing.assert_allclose(out[:, 0], 120.0 / 360.0, atol=1e-4)

    def test_pure_blue_hue(self):
        """純青 (0,0,1) の H ≈ 240/360 ≈ 0.667."""
        p = PROJECT_ROOT / "models" / CATEGORY / "hsv_h.onnx"
        sess = ort.InferenceSession(str(p))
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        img[:, 2] = 1.0  # B=1
        out = _run(sess, img)
        np.testing.assert_allclose(out[:, 0], 240.0 / 360.0, atol=1e-4)

    def test_grayscale_saturation_zero(self):
        """無彩色画像の S = 0."""
        p = PROJECT_ROOT / "models" / CATEGORY / "hsv_s.onnx"
        sess = ort.InferenceSession(str(p))
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_value_is_max_channel(self):
        """V = max(R, G, B) であること."""
        p = PROJECT_ROOT / "models" / CATEGORY / "hsv_v.onnx"
        sess = ort.InferenceSession(str(p))
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 8, 8), dtype=np.float32)
        out = _run(sess, img)
        expected = img.max(axis=1, keepdims=True)
        np.testing.assert_allclose(out[:, 0:1], expected, atol=1e-6)
