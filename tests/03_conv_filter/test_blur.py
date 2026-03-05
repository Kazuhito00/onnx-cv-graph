"""平均ぼかし (box filter) モデルのテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
3×3 / 5×5 / 7×7 の全バリアントをテストする.
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

KERNEL_SIZES = [3, 5, 7]


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    missing = any(
        not (MODEL_DIR / f"blur_{k}x{k}.onnx").exists() for k in KERNEL_SIZES
    )
    if missing:
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )
    for k in KERNEL_SIZES:
        p = MODEL_DIR / f"blur_{k}x{k}.onnx"
        assert p.exists(), f"モデルが見つかりません: {p}"


@pytest.fixture(scope="module", params=KERNEL_SIZES, ids=[f"{k}x{k}" for k in KERNEL_SIZES])
def kernel_size(request):
    """テスト対象のカーネルサイズ."""
    return request.param


@pytest.fixture(scope="module", params=KERNEL_SIZES, ids=[f"sess_{k}x{k}" for k in KERNEL_SIZES])
def session_and_k(request):
    """(session, kernel_size) のタプルを返す."""
    k = request.param
    sess = ort.InferenceSession(str(MODEL_DIR / f"blur_{k}x{k}.onnx"))
    return sess, k


def _run(session, img: np.ndarray) -> np.ndarray:
    """セッションで推論を実行し、最初の出力を返すヘルパー."""
    return session.run(None, {"input": img})[0]


def _numpy_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """NumPy 参照実装: reflect パディング + チャネルごと平均フィルタ.

    img: (N, 3, H, W) float32
    """
    n, c, h, w = img.shape
    pad = kernel_size // 2
    result = np.zeros_like(img)
    for b in range(n):
        for ch in range(c):
            # reflect パディング
            padded = np.pad(img[b, ch], pad, mode="reflect")
            # 平均フィルタ (スライディングウィンドウ)
            out = np.zeros((h, w), dtype=np.float32)
            for dy in range(kernel_size):
                for dx in range(kernel_size):
                    out += padded[dy:dy + h, dx:dx + w]
            result[b, ch] = out / (kernel_size * kernel_size)
    return result


class TestBlurOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session_and_k):
        """単一画像 (N=1) で出力が (1,3,H,W) になること."""
        sess, k = session_and_k
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session_and_k):
        """バッチ (N>1) で出力が (N,3,H,W) になること."""
        sess, k = session_and_k
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (2, 3, 8, 8)

    def test_rectangular(self, session_and_k):
        """非正方形画像で出力形状が入力と同じであること."""
        sess, k = session_and_k
        img = np.random.rand(1, 3, 12, 20).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (1, 3, 12, 20)


class TestBlurValues:
    """出力値の正確性を検証するテスト群."""

    def test_uniform_image_unchanged(self, session_and_k):
        """均一画像はぼかし後も変化しないこと."""
        sess, k = session_and_k
        val = 0.6
        img = np.full((1, 3, 16, 16), val, dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_output_range(self, session_and_k):
        """出力値が [0, 1] 範囲内であること."""
        sess, k = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(sess, img)
        assert out.min() >= -1e-6, f"出力に負の値: {out.min()}"
        assert out.max() <= 1.0 + 1e-6, f"出力が 1.0 を超過: {out.max()}"

    def test_matches_numpy_reference(self, session_and_k):
        """ランダム入力で NumPy 参照実装と一致すること."""
        sess, k = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        expected = _numpy_blur(img, k)
        out = _run(sess, img)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_all_black(self, session_and_k):
        """全黒画像のぼかし結果が全黒であること."""
        sess, k = session_and_k
        img = np.zeros((1, 3, 8, 8), dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_array_equal(out, img)

    def test_all_white(self, session_and_k):
        """全白画像のぼかし結果が全白であること."""
        sess, k = session_and_k
        img = np.ones((1, 3, 8, 8), dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_channels_independent(self, session_and_k):
        """各チャネルが独立してぼかされること."""
        sess, k = session_and_k
        rng = np.random.default_rng(77)
        # R チャネルのみ非ゼロ
        img = np.zeros((1, 3, 16, 16), dtype=np.float32)
        img[0, 0] = rng.random((16, 16), dtype=np.float32)
        out = _run(sess, img)
        # G, B チャネルはゼロのまま
        np.testing.assert_allclose(out[0, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(out[0, 2], 0.0, atol=1e-6)
        # R チャネルは非ゼロ
        assert out[0, 0].sum() > 0


class TestBlurVsOpenCV:
    """OpenCV の blur (boxFilter) との比較テスト群.

    ONNX は float32 で演算するため、OpenCV も float32 で比較する.
    パディングモードの違い (ONNX reflect = BORDER_REFLECT) を合わせて比較.
    uint8 変換後は ±1 以内の誤差を許容する.
    """

    def test_random_image_float32(self, session_and_k):
        """ランダム画像で OpenCV blur (float32) との一致を検証."""
        sess, k = session_and_k
        rng = np.random.default_rng(123)
        img_nchw = rng.random((1, 3, 32, 32), dtype=np.float32)

        # ONNX 推論
        onnx_out = _run(sess, img_nchw)

        # OpenCV: float32 HWC で blur を適用 (borderType を ONNX の reflect に合わせる)
        hwc_f32 = img_nchw[0].transpose(1, 2, 0)  # (H, W, 3)
        cv_out = cv2.blur(hwc_f32, (k, k), borderType=cv2.BORDER_REFLECT_101)
        cv_nchw = cv_out.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        np.testing.assert_allclose(onnx_out, cv_nchw, atol=1e-5)

    def test_random_image_uint8(self, session_and_k):
        """uint8 変換後の比較 (±1 許容)."""
        sess, k = session_and_k
        rng = np.random.default_rng(456)
        img_nchw = rng.random((1, 3, 32, 32), dtype=np.float32)

        onnx_out = _run(sess, img_nchw)
        onnx_uint8 = (onnx_out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)

        # OpenCV float32 で blur → uint8 変換
        hwc_f32 = img_nchw[0].transpose(1, 2, 0)
        cv_out = cv2.blur(hwc_f32, (k, k), borderType=cv2.BORDER_REFLECT_101)
        cv_uint8 = (cv_out * 255.0).clip(0, 255).round().astype(np.uint8)

        diff = np.abs(onnx_uint8.astype(np.int16) - cv_uint8.astype(np.int16))
        assert diff.max() <= 1, f"最大誤差 {diff.max()} > 1 (kernel={k}x{k})"
