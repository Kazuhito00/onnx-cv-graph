"""Laplacian of Gaussian (LoG) モデルのテスト."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
KERNEL_SIZES = [5, 7]


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    paths = [PROJECT_ROOT / "models" / CATEGORY / f"log_{k}x{k}.onnx" for k in KERNEL_SIZES]
    if not all(p.exists() for p in paths):
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module", params=KERNEL_SIZES)
def session_and_k(request):
    k = request.param
    path = PROJECT_ROOT / "models" / CATEGORY / f"log_{k}x{k}.onnx"
    return ort.InferenceSession(str(path)), k


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


class TestLogFilterOutputShape:
    def test_shape_preserved(self, session_and_k):
        session, _ = session_and_k
        assert _run(session, np.random.rand(1, 3, 16, 16).astype(np.float32)).shape == (1, 3, 16, 16)

    def test_batch(self, session_and_k):
        session, _ = session_and_k
        assert _run(session, np.random.rand(2, 3, 8, 8).astype(np.float32)).shape == (2, 3, 8, 8)


class TestLogFilterValues:
    def test_uniform_is_zero(self, session_and_k):
        session, _ = session_and_k
        img = np.full((1, 3, 16, 16), 0.5, dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-4)

    def test_output_range(self, session_and_k):
        session, _ = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_3ch_identical(self, session_and_k):
        session, _ = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])

    def test_edge_detected(self, session_and_k):
        session, _ = session_and_k
        img = np.zeros((1, 3, 16, 16), dtype=np.float32)
        img[:, :, :, 8:] = 1.0
        out = _run(session, img)
        edge_col = out[0, 0, :, 7:9].mean()
        flat_col = out[0, 0, :, 0:3].mean()
        assert edge_col > flat_col
