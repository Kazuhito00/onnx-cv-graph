"""シャープ化モデルのテスト."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "sharpen.onnx"


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


class TestSharpenOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        assert _run(session, img).shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        assert _run(session, img).shape == (2, 3, 8, 8)


class TestSharpenValues:
    def test_uniform_unchanged(self, session):
        """均一画像はシャープ化しても変わらない."""
        img = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_output_range(self, session):
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_enhances_edges(self, session):
        """エッジ部分がシャープ化で強調される."""
        img = np.zeros((1, 3, 8, 8), dtype=np.float32)
        img[:, :, :, 4:] = 1.0  # 右半分が白
        out = _run(session, img)
        # エッジ付近で元画像よりオーバーシュートまたはアンダーシュート
        # (Clip で 0/1 に制限されるが、エッジ部分のコントラストは維持)
        assert out[:, :, :, 0].mean() < 0.01  # 黒部分は黒のまま
        assert out[:, :, :, 7].mean() > 0.99  # 白部分は白のまま

    def test_matches_numpy(self, session):
        """NumPy 参照実装と一致."""
        correlate = pytest.importorskip("scipy.ndimage").correlate
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        expected = np.zeros_like(img)
        for n in range(img.shape[0]):
            for c in range(img.shape[1]):
                expected[n, c] = correlate(img[n, c], k, mode='mirror')
        expected = np.clip(expected, 0, 1)
        np.testing.assert_allclose(out, expected, atol=1e-5)
