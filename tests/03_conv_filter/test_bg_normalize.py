"""背景ムラ補正モデルのテスト.

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
MODEL_DIR = PROJECT_ROOT / "models" / CATEGORY

KERNEL_SIZES = [3, 5, 7]


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    missing = any(
        not (MODEL_DIR / f"bg_normalize_{k}x{k}.onnx").exists() for k in KERNEL_SIZES
    )
    if missing:
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module", params=KERNEL_SIZES, ids=[f"sess_{k}x{k}" for k in KERNEL_SIZES])
def session_and_k(request):
    k = request.param
    sess = ort.InferenceSession(str(MODEL_DIR / f"bg_normalize_{k}x{k}.onnx"))
    return sess, k


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


def _gaussian_kernel_2d(ksize: int) -> np.ndarray:
    """テスト用ガウシアンカーネル生成 (OpenCV デフォルト σ)."""
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    ax = np.arange(ksize, dtype=np.float32) - (ksize - 1) / 2.0
    k1d = np.exp(-0.5 * (ax / sigma) ** 2)
    k1d /= k1d.sum()
    return np.outer(k1d, k1d).astype(np.float32)


class TestBgNormalizeOutputShape:
    def test_single_image(self, session_and_k):
        sess, k = session_and_k
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session_and_k):
        sess, k = session_and_k
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (2, 3, 8, 8)


class TestBgNormalizeValues:
    def test_uniform_image_mid_gray(self, session_and_k):
        """均一画像: blur == input なので diff=0 → 0+0.5=0.5."""
        sess, k = session_and_k
        img = np.full((1, 3, 16, 16), 0.3, dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_allclose(out, 0.5, atol=1e-4)

    def test_shadow_removal(self, session_and_k):
        """明暗グラデーション上のテキスト → 出力が均一化されること."""
        sess, k = session_and_k
        h, w = 32, 32
        # 左から右に暗くなるグラデーション (影を模擬)
        grad = np.linspace(0.8, 0.3, w, dtype=np.float32)
        bg = np.broadcast_to(grad, (1, 3, h, w)).copy()
        # テキスト: 中央に暗い線 (文字を模擬)
        img = bg.copy()
        img[:, :, 14:18, 10:22] = np.clip(bg[:, :, 14:18, 10:22] - 0.3, 0, 1)

        out = _run(sess, img)

        # テキスト部分の出力値が左右でほぼ均一になること
        text_left = out[0, 0, 16, 11]
        text_right = out[0, 0, 16, 20]
        assert abs(text_left - text_right) < 0.15, (
            f"ムラ補正後もテキスト部分の明暗差が大きい: left={text_left:.3f}, right={text_right:.3f}"
        )

    def test_output_range(self, session_and_k):
        """出力値が [0, 1] 範囲内."""
        sess, k = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(sess, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_high_frequency_preserved(self, session_and_k):
        """高周波成分 (エッジ) が保持されること."""
        sess, k = session_and_k
        img = np.full((1, 3, 32, 32), 0.5, dtype=np.float32)
        # シャープなエッジ
        img[:, :, :, 16:] = 0.8
        out = _run(sess, img)
        # エッジ付近でコントラストが残っていること
        edge_diff = abs(float(out[0, 0, 16, 15]) - float(out[0, 0, 16, 16]))
        assert edge_diff > 0.05


class TestBgNormalizeVsScipy:
    """scipy を使った参照実装との比較."""

    def test_matches_scipy_reference(self, session_and_k):
        correlate = pytest.importorskip("scipy.ndimage").correlate

        sess, k = session_and_k
        g2d = _gaussian_kernel_2d(k)

        rng = np.random.default_rng(123)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)

        # scipy 参照実装
        blur = np.zeros_like(img)
        for c in range(3):
            blur[0, c] = correlate(img[0, c], g2d, mode="mirror")
        expected = np.clip(img - blur + 0.5, 0.0, 1.0).astype(np.float32)

        out = _run(sess, img)
        np.testing.assert_allclose(out, expected, atol=1e-5)
