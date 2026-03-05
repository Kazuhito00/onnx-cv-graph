"""ゼロ化閾値処理モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "thresh_tozero.onnx"

LUMA_WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    if not MODEL_PATH.exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )
    assert MODEL_PATH.exists(), f"モデルが見つかりません: {MODEL_PATH}"


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_PATH))


def _run(session, img: np.ndarray, threshold: float) -> np.ndarray:
    thr = np.array([threshold], dtype=np.float32)
    return session.run(None, {"input": img, "threshold": thr})[0]


def _numpy_thresh_tozero(img: np.ndarray, threshold: float) -> np.ndarray:
    """NumPy 参照実装."""
    gray = (img * LUMA_WEIGHTS).sum(axis=1, keepdims=True)
    mask = (gray > threshold).astype(np.float32)
    result = gray * mask
    return np.repeat(result, 3, axis=1)


class TestThreshTozeroOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.5)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.5)
        assert out.shape == (2, 3, 8, 8)


class TestThreshTozeroValues:
    def test_dark_pixels_zeroed(self, session):
        """閾値以下の画素が 0 になること."""
        img = np.full((1, 3, 4, 4), 0.2, dtype=np.float32)
        out = _run(session, img, 0.5)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_bright_pixels_kept(self, session):
        """閾値超過の画素が gray 値を維持すること."""
        img = np.full((1, 3, 4, 4), 0.8, dtype=np.float32)
        out = _run(session, img, 0.5)
        np.testing.assert_allclose(out, 0.8, atol=1e-4)

    def test_all_black(self, session):
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img, 0.5)
        np.testing.assert_array_equal(out, 0.0)

    def test_matches_numpy_reference(self, session):
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        expected = _numpy_thresh_tozero(img, 0.4)
        out = _run(session, img, 0.4)
        np.testing.assert_allclose(out, expected, atol=1e-5)


class TestThreshTozeroVsOpenCV:
    def test_random_image(self, session):
        rng = np.random.default_rng(123)
        img_nchw = rng.random((1, 3, 32, 32), dtype=np.float32)
        threshold = 0.5

        onnx_out = _run(session, img_nchw, threshold)
        onnx_uint8 = (onnx_out[0, 0] * 255.0).clip(0, 255).round().astype(np.uint8)

        hwc_rgb = (img_nchw[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        hwc_bgr = cv2.cvtColor(hwc_rgb, cv2.COLOR_RGB2BGR)
        cv_gray = cv2.cvtColor(hwc_bgr, cv2.COLOR_BGR2GRAY)
        thr_uint8 = int(threshold * 255)
        _, cv_tozero = cv2.threshold(cv_gray, thr_uint8, 255, cv2.THRESH_TOZERO)

        # 閾値境界付近を除外して比較 (量子化誤差)
        non_boundary = np.abs(cv_gray.astype(np.int16) - thr_uint8) > 2
        assert non_boundary.sum() > 0
        np.testing.assert_allclose(
            onnx_uint8[non_boundary], cv_tozero[non_boundary], atol=2)
