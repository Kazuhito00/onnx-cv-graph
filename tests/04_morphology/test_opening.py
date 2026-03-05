"""オープニング (opening) モデルのテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
3×3 / 5×5 の全バリアントをテストする.
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
MODEL_DIR = PROJECT_ROOT / "models" / CATEGORY

KERNEL_SIZES = [3, 5]


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    missing = any(
        not (MODEL_DIR / f"opening_{k}x{k}.onnx").exists() for k in KERNEL_SIZES
    )
    if missing:
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module", params=KERNEL_SIZES, ids=[f"sess_{k}x{k}" for k in KERNEL_SIZES])
def session_and_k(request):
    """(session, kernel_size) のタプルを返す."""
    k = request.param
    sess = ort.InferenceSession(str(MODEL_DIR / f"opening_{k}x{k}.onnx"))
    return sess, k


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


class TestOpeningOutputShape:
    """出力テンソルの形状を検証するテスト群."""

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


class TestOpeningValues:
    """出力値の正確性を検証するテスト群."""

    def test_uniform_image_unchanged(self, session_and_k):
        """均一画像はオープニング後も変化しないこと."""
        sess, k = session_and_k
        img = np.full((1, 3, 16, 16), 0.5, dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_all_black(self, session_and_k):
        sess, k = session_and_k
        img = np.zeros((1, 3, 8, 8), dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_array_equal(out, img)

    def test_all_white(self, session_and_k):
        sess, k = session_and_k
        img = np.ones((1, 3, 8, 8), dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_removes_small_bright_noise(self, session_and_k):
        """小さい明るいノイズが除去されること (opening = erode → dilate)."""
        sess, k = session_and_k
        img = np.zeros((1, 3, 16, 16), dtype=np.float32)
        img[:, :, 8, 8] = 1.0  # 1ピクセルの明るいノイズ
        out = _run(sess, img)
        # 収縮で消え、膨張で復元されないのでゼロのまま
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_output_le_input(self, session_and_k):
        """オープニングは常に入力以下の値を返すこと (anti-extensive)."""
        sess, k = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(sess, img)
        assert (out <= img + 1e-6).all()


class TestOpeningVsOpenCV:
    """OpenCV の morphologyEx(MORPH_OPEN) との比較テスト群."""

    def test_matches_opencv(self, session_and_k):
        """ランダム画像で OpenCV opening との一致を検証 (uint8 ±1 許容)."""
        sess, k = session_and_k
        rng = np.random.default_rng(123)
        img_nchw = rng.random((1, 3, 32, 32), dtype=np.float32)

        onnx_out = _run(sess, img_nchw)
        onnx_uint8 = (onnx_out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)

        hwc_u8 = (img_nchw[0].transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)
        kernel = np.ones((k, k), dtype=np.uint8)
        cv_out = cv2.morphologyEx(hwc_u8, cv2.MORPH_OPEN, kernel)

        diff = np.abs(onnx_uint8.astype(np.int16) - cv_out.astype(np.int16))
        assert diff.max() <= 1, f"最大誤差 {diff.max()} > 1 (kernel={k}x{k})"
