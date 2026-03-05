"""ガウシアンぼかしモデルのテスト."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
KERNEL_SIZES = [3, 5, 7]


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    paths = [PROJECT_ROOT / "models" / CATEGORY / f"gaussian_blur_{k}x{k}.onnx" for k in KERNEL_SIZES]
    if not all(p.exists() for p in paths):
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )
    for p in paths:
        assert p.exists()


@pytest.fixture(scope="module", params=KERNEL_SIZES)
def session_and_k(request):
    k = request.param
    path = PROJECT_ROOT / "models" / CATEGORY / f"gaussian_blur_{k}x{k}.onnx"
    return ort.InferenceSession(str(path)), k


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


def _gaussian_kernel_1d(ksize):
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    ax = np.arange(ksize, dtype=np.float64) - (ksize - 1) / 2.0
    kernel = np.exp(-0.5 * (ax / sigma) ** 2)
    return kernel / kernel.sum()


def _numpy_gaussian_blur(img, k):
    from scipy.ndimage import correlate1d
    k1d = _gaussian_kernel_1d(k)
    out = img.copy()
    for n in range(img.shape[0]):
        for c in range(img.shape[1]):
            tmp = correlate1d(out[n, c], k1d, axis=0, mode='mirror')
            out[n, c] = correlate1d(tmp, k1d, axis=1, mode='mirror')
    return out.astype(np.float32)


class TestGaussianBlurOutputShape:
    def test_shape_preserved(self, session_and_k):
        session, _ = session_and_k
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        assert _run(session, img).shape == (1, 3, 16, 16)

    def test_batch(self, session_and_k):
        session, _ = session_and_k
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        assert _run(session, img).shape == (2, 3, 8, 8)


class TestGaussianBlurValues:
    def test_uniform_unchanged(self, session_and_k):
        """均一画像はぼかしても変わらない."""
        session, _ = session_and_k
        img = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_output_range(self, session_and_k):
        session, _ = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_matches_numpy(self, session_and_k):
        """NumPy (scipy) 参照実装と一致."""
        pytest.importorskip("scipy")
        session, k = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        expected = _numpy_gaussian_blur(img, k)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_channel_independence(self, session_and_k):
        """各チャネルが独立に処理される."""
        session, _ = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        img[:, 1:] = 0.0
        out = _run(session, img)
        np.testing.assert_allclose(out[:, 1:], 0.0, atol=1e-6)


class TestGaussianBlurVsOpenCV:
    def test_vs_opencv(self, session_and_k):
        cv2 = pytest.importorskip("cv2")
        session, k = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 32, 32), dtype=np.float32)
        out = _run(session, img)
        # OpenCV: HWC, BGR
        hwc = img[0].transpose(1, 2, 0)
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        cv_out = cv2.GaussianBlur(hwc, (k, k), sigma, borderType=cv2.BORDER_REFLECT_101)
        if cv_out.ndim == 2:
            cv_out = cv_out[:, :, None]
        expected = cv_out.transpose(2, 0, 1)[None]
        np.testing.assert_allclose(out, expected, atol=1e-5)
