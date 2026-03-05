"""切り詰め閾値処理モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "thresh_trunc.onnx"

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


def _numpy_thresh_trunc(img: np.ndarray, threshold: float) -> np.ndarray:
    """NumPy 参照実装."""
    gray = (img * LUMA_WEIGHTS).sum(axis=1, keepdims=True)
    trunc = np.minimum(gray, threshold)
    return np.repeat(trunc, 3, axis=1)


class TestThreshTruncOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.5)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.5)
        assert out.shape == (2, 3, 8, 8)


class TestThreshTruncValues:
    def test_output_le_threshold(self, session):
        """出力値が threshold 以下であること."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        thr = 0.6
        out = _run(session, img, thr)
        assert out.max() <= thr + 1e-6

    def test_dark_pixels_unchanged(self, session):
        """閾値以下の画素は元の gray 値を維持すること."""
        img = np.full((1, 3, 4, 4), 0.2, dtype=np.float32)
        out = _run(session, img, 0.5)
        expected_gray = (0.2 * LUMA_WEIGHTS).sum()  # 重み合計が厳密に1でない
        np.testing.assert_allclose(out, expected_gray, atol=1e-4)

    def test_bright_pixels_truncated(self, session):
        """閾値超過の画素は threshold に切り詰められること."""
        img = np.full((1, 3, 4, 4), 0.8, dtype=np.float32)
        out = _run(session, img, 0.5)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_matches_numpy_reference(self, session):
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        expected = _numpy_thresh_trunc(img, 0.4)
        out = _run(session, img, 0.4)
        np.testing.assert_allclose(out, expected, atol=1e-5)


class TestThreshTruncVsOpenCV:
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
        _, cv_trunc = cv2.threshold(cv_gray, thr_uint8, 255, cv2.THRESH_TRUNC)

        # uint8 量子化誤差を許容
        np.testing.assert_allclose(onnx_uint8, cv_trunc, atol=2)
