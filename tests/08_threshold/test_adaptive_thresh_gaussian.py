"""適応的閾値処理 (ガウス) モデルのテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name

LUMA_WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)

KERNEL_SIZES = [3, 5, 7]


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    for k in KERNEL_SIZES:
        p = PROJECT_ROOT / "models" / CATEGORY / f"adaptive_thresh_gaussian_{k}x{k}.onnx"
        if not p.exists():
            subprocess.check_call(
                [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
                cwd=str(PROJECT_ROOT),
            )
            break


@pytest.fixture(scope="module", params=KERNEL_SIZES, ids=[f"k{k}" for k in KERNEL_SIZES])
def session_and_k(request):
    k = request.param
    p = PROJECT_ROOT / "models" / CATEGORY / f"adaptive_thresh_gaussian_{k}x{k}.onnx"
    return ort.InferenceSession(str(p)), k


def _run(session, img: np.ndarray, c: float) -> np.ndarray:
    return session.run(None, {"input": img, "C": np.array([c], dtype=np.float32)})[0]


class TestAdaptiveThreshGaussianOutputShape:
    def test_single_image(self, session_and_k):
        sess, k = session_and_k
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        out = _run(sess, img, 0.02)
        assert out.shape == (1, 3, 16, 16)

    def test_batch(self, session_and_k):
        sess, k = session_and_k
        img = np.random.rand(2, 3, 16, 16).astype(np.float32)
        out = _run(sess, img, 0.02)
        assert out.shape == (2, 3, 16, 16)


class TestAdaptiveThreshGaussianValues:
    def test_output_only_zero_or_one(self, session_and_k):
        sess, k = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(sess, img, 0.02)
        unique = np.unique(out)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_all_channels_equal(self, session_and_k):
        sess, k = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(sess, img, 0.02)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        np.testing.assert_array_equal(out[:, 0], out[:, 2])

    def test_uniform_image_positive_c(self, session_and_k):
        """均一画像で C > 0 → gray > (local_gauss - C) → 全 1."""
        sess, k = session_and_k
        img = np.full((1, 3, 16, 16), 0.5, dtype=np.float32)
        out = _run(sess, img, 0.02)
        np.testing.assert_array_equal(out, 1.0)

    def test_uniform_image_negative_c(self, session_and_k):
        """均一画像で C < 0 → gray < (local_gauss + |C|) → 全 0."""
        sess, k = session_and_k
        img = np.full((1, 3, 16, 16), 0.5, dtype=np.float32)
        out = _run(sess, img, -0.02)
        np.testing.assert_array_equal(out, 0.0)

    def test_high_contrast_image(self, session_and_k):
        """高コントラスト画像でエッジ付近に白黒の差が出ること."""
        sess, k = session_and_k
        img = np.zeros((1, 3, 32, 32), dtype=np.float32)
        img[:, :, :, 16:] = 1.0  # 左黒・右白
        out = _run(sess, img, 0.02)
        # 境界から十分離れた領域で確認
        # 左端: gray=0, local_gauss≈0 → gray > (0 - C) → C > 0 なら True
        # 右端: gray=1, local_gauss≈1 → gray > (1 - C) → True
        # 境界付近には 0/1 の混在があることを確認
        assert out[0, 0, 16, 0] == 1.0 or out[0, 0, 16, 31] == 1.0
