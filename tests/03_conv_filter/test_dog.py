"""DoG (Difference of Gaussians) モデルのテスト."""

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
    if not (MODEL_DIR / "dog_3x3_5x5.onnx").exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module", params=["dog_3x3_5x5", "dog_3x3_7x7", "dog_5x5_7x7"])
def session(request):
    return ort.InferenceSession(str(MODEL_DIR / f"{request.param}.onnx"))


@pytest.fixture(scope="module")
def session_3x3_5x5():
    return ort.InferenceSession(str(MODEL_DIR / "dog_3x3_5x5.onnx"))


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


class TestDogOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        assert _run(session, img).shape == (1, 3, 16, 16)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 16, 16).astype(np.float32)
        assert _run(session, img).shape == (2, 3, 16, 16)


class TestDogValues:
    def test_uniform_is_zero(self, session):
        """均一画像の DoG は 0 (差分なし)."""
        img = np.full((1, 3, 16, 16), 0.5, dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_output_range(self, session):
        """出力が [0, 1] 範囲内."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 32, 32), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_3ch_identical(self, session):
        """出力 3 チャネルが同一."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        np.testing.assert_array_equal(out[:, 0], out[:, 2])

    def test_edge_detected(self, session):
        """エッジ部分で高い値が検出される."""
        img = np.zeros((1, 3, 32, 32), dtype=np.float32)
        img[:, :, :, 16:] = 1.0
        out = _run(session, img)
        edge_region = out[0, 0, :, 14:18].mean()
        flat_region = out[0, 0, :, 0:4].mean()
        assert edge_region > flat_region


class TestDogVsOpenCV:
    def test_vs_opencv(self, session_3x3_5x5):
        """OpenCV の GaussianBlur 差分と比較."""
        cv2 = pytest.importorskip("cv2")
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 32, 32), dtype=np.float32)
        out = _run(session_3x3_5x5, img)

        # OpenCV で参照実装
        hwc = img[0].transpose(1, 2, 0)
        gray = cv2.cvtColor(hwc, cv2.COLOR_RGB2GRAY)
        # OpenCV デフォルト σ
        sigma1 = 0.3 * ((3 - 1) * 0.5 - 1) + 0.8
        sigma2 = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
        blur1 = cv2.GaussianBlur(gray, (3, 3), sigma1, borderType=cv2.BORDER_REFLECT_101)
        blur2 = cv2.GaussianBlur(gray, (5, 5), sigma2, borderType=cv2.BORDER_REFLECT_101)
        expected = np.clip(np.abs(blur1 - blur2) / 0.25, 0, 1)

        np.testing.assert_allclose(out[0, 0], expected, atol=0.05)
