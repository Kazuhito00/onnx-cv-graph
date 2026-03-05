"""Sobel エッジ検出モデルのテスト."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "sobel.onnx"


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


class TestSobelOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        assert _run(session, img).shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        assert _run(session, img).shape == (2, 3, 8, 8)


class TestSobelValues:
    def test_uniform_is_zero(self, session):
        """均一画像のエッジは 0."""
        img = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_output_range(self, session):
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_3ch_identical(self, session):
        """出力3チャネルが同一."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        np.testing.assert_array_equal(out[:, 0], out[:, 2])

    def test_vertical_edge_detected(self, session):
        """垂直エッジが検出される."""
        img = np.zeros((1, 3, 8, 8), dtype=np.float32)
        img[:, :, :, 4:] = 1.0
        out = _run(session, img)
        # エッジ列 (col 3,4 付近) で高い値
        edge_region = out[0, 0, :, 3:5].mean()
        flat_region = out[0, 0, :, 0:2].mean()
        assert edge_region > flat_region

    def test_horizontal_edge_detected(self, session):
        """水平エッジが検出される."""
        img = np.zeros((1, 3, 8, 8), dtype=np.float32)
        img[:, :, 4:, :] = 1.0
        out = _run(session, img)
        edge_region = out[0, 0, 3:5, :].mean()
        flat_region = out[0, 0, 0:2, :].mean()
        assert edge_region > flat_region

    def test_vs_opencv(self, session):
        cv2 = pytest.importorskip("cv2")
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 32, 32), dtype=np.float32)
        out = _run(session, img)
        # OpenCV: グレースケール化 → Sobel
        hwc = img[0].transpose(1, 2, 0)
        gray = cv2.cvtColor(hwc, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
        expected = np.clip(np.abs(gx) + np.abs(gy), 0, 1)
        # 輝度重みの違いにより多少の差が出るため atol を緩める
        np.testing.assert_allclose(out[0, 0], expected, atol=0.05)
