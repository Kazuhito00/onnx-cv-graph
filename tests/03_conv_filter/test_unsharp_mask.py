"""アンシャープマスクモデルのテスト."""

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
    paths = [PROJECT_ROOT / "models" / CATEGORY / f"unsharp_mask_{k}x{k}.onnx" for k in KERNEL_SIZES]
    if not all(p.exists() for p in paths):
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module", params=KERNEL_SIZES)
def session_and_k(request):
    k = request.param
    path = PROJECT_ROOT / "models" / CATEGORY / f"unsharp_mask_{k}x{k}.onnx"
    return ort.InferenceSession(str(path)), k


def _run(session, img: np.ndarray, amount: float) -> np.ndarray:
    return session.run(None, {
        "input": img,
        "amount": np.array([amount], dtype=np.float32),
    })[0]


class TestUnsharpMaskOutputShape:
    def test_shape_preserved(self, session_and_k):
        session, _ = session_and_k
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        assert _run(session, img, 1.0).shape == (1, 3, 16, 16)

    def test_batch(self, session_and_k):
        session, _ = session_and_k
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        assert _run(session, img, 1.0).shape == (2, 3, 8, 8)


class TestUnsharpMaskValues:
    def test_amount_zero_is_identity(self, session_and_k):
        """amount=0 で元画像と同一."""
        session, _ = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img, 0.0)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_uniform_unchanged(self, session_and_k):
        session, _ = session_and_k
        img = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)
        out = _run(session, img, 2.0)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_output_range(self, session_and_k):
        session, _ = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img, 3.0)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_higher_amount_sharper(self, session_and_k):
        """amount が大きいほどシャープ."""
        session, _ = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out1 = _run(session, img, 1.0)
        out3 = _run(session, img, 3.0)
        # 差分の標準偏差が大きいほどコントラスト増幅
        diff1 = np.std(out1 - img)
        diff3 = np.std(out3 - img)
        assert diff3 > diff1
