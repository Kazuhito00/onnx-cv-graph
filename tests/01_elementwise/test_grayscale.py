"""グレースケール変換モデルのテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "grayscale.onnx"


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    if not MODEL_PATH.exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )
    assert MODEL_PATH.exists(), f"モデルが見つかりません: {MODEL_PATH}"


@pytest.fixture(scope="module")
def session():
    """ONNX Runtime の推論セッションを返す."""
    return ort.InferenceSession(str(MODEL_PATH))


def _run(session, img: np.ndarray) -> np.ndarray:
    """セッションで推論を実行し、最初の出力を返すヘルパー."""
    return session.run(None, {"input": img})[0]


class TestGrayscaleOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        """単一画像 (N=1) で出力が (1,3,H,W) になること."""
        img = np.random.rand(1, 3, 4, 4).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 4, 4)

    def test_batch(self, session):
        """バッチ (N>1) で出力が (N,3,H,W) になること."""
        img = np.random.rand(3, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (3, 3, 8, 8)

    def test_all_channels_equal(self, session):
        """出力の3チャネルが全て同一値であること."""
        rng = np.random.default_rng(99)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        np.testing.assert_array_equal(out[:, 0], out[:, 2])


class TestGrayscaleValues:
    """出力値の正確性を検証するテスト群."""

    def test_pure_red(self, session):
        """純赤 (1,0,0) → R 重み 0.2989 に一致すること."""
        img = np.zeros((1, 3, 1, 1), dtype=np.float32)
        img[:, 0, :, :] = 1.0
        out = _run(session, img)
        # 3ch 全て同一値
        np.testing.assert_allclose(out[0, 0, 0, 0], 0.2989, atol=1e-5)
        np.testing.assert_allclose(out[0, 1, 0, 0], 0.2989, atol=1e-5)
        np.testing.assert_allclose(out[0, 2, 0, 0], 0.2989, atol=1e-5)

    def test_pure_white(self, session):
        """純白 (1,1,1) → 全重み合計 ≈ 1.0 に一致すること."""
        img = np.ones((1, 3, 1, 1), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 1.0, atol=1e-4)

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
        gray_1ch = (img * weights).sum(axis=1, keepdims=True)
        expected = np.repeat(gray_1ch, 3, axis=1)
        out = _run(session, img)
        np.testing.assert_allclose(out, expected, atol=1e-5)


def _nchw_to_hwc_uint8(nchw: np.ndarray) -> np.ndarray:
    """NCHW float32 [0,1] → HWC uint8 [0,255] に変換するヘルパー (単一画像)."""
    return (nchw[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)


class TestGrayscaleVsOpenCV:
    """OpenCV の cvtColor(BGR2GRAY) との比較テスト群.

    OpenCV は BGR 順・HWC・uint8 で動作し、内部で四捨五入を行うため
    uint8 量子化誤差が生じる. そのため ONNX 出力を uint8 に変換した上で
    許容誤差 1 以内で比較する.
    """

    def test_random_image(self, session):
        """ランダム画像で OpenCV cvtColor との差が uint8 で ±1 以内であること."""
        rng = np.random.default_rng(123)
        # NCHW float32 (RGB 順)
        img_nchw = rng.random((1, 3, 32, 32), dtype=np.float32)

        # ONNX 推論 → (1,3,H,W) から ch0 を取得 (3ch同一値)
        onnx_out = _run(session, img_nchw)
        onnx_gray_uint8 = (onnx_out[0, 0] * 255.0).clip(0, 255).round().astype(np.uint8)

        # OpenCV: RGB→BGR に変換してから cvtColor
        hwc_rgb = _nchw_to_hwc_uint8(img_nchw)
        hwc_bgr = cv2.cvtColor(hwc_rgb, cv2.COLOR_RGB2BGR)
        cv_gray = cv2.cvtColor(hwc_bgr, cv2.COLOR_BGR2GRAY)

        # uint8 量子化誤差を考慮し ±1 以内で比較
        np.testing.assert_allclose(
            onnx_gray_uint8.astype(np.int16),
            cv_gray.astype(np.int16),
            atol=1,
        )

    def test_pure_colors(self, session):
        """純色 (赤/緑/青) で OpenCV と一致すること."""
        for ch, name in [(0, "赤"), (1, "緑"), (2, "青")]:
            img_nchw = np.zeros((1, 3, 1, 1), dtype=np.float32)
            img_nchw[:, ch, :, :] = 1.0

            onnx_out = _run(session, img_nchw)
            onnx_val = (onnx_out[0, 0, 0, 0] * 255.0).clip(0, 255).round().astype(np.uint8)

            hwc_rgb = _nchw_to_hwc_uint8(img_nchw)
            hwc_bgr = cv2.cvtColor(hwc_rgb, cv2.COLOR_RGB2BGR)
            cv_val = cv2.cvtColor(hwc_bgr, cv2.COLOR_BGR2GRAY)[0, 0]

            assert abs(int(onnx_val) - int(cv_val)) <= 1, (
                f"{name}チャネル: ONNX={onnx_val}, OpenCV={cv_val}"
            )

    def test_gradient_image(self, session):
        """水平グラデーション画像で OpenCV との最大差が ±1 以内であること."""
        # 幅方向に 0→255 のグラデーション (3ch 同値 = グレー階調)
        grad = np.linspace(0, 1, 256, dtype=np.float32)
        img_nchw = np.zeros((1, 3, 1, 256), dtype=np.float32)
        for c in range(3):
            img_nchw[0, c, 0, :] = grad

        onnx_out = _run(session, img_nchw)
        onnx_gray_uint8 = (onnx_out[0, 0] * 255.0).clip(0, 255).round().astype(np.uint8)

        hwc_rgb = _nchw_to_hwc_uint8(img_nchw)
        hwc_bgr = cv2.cvtColor(hwc_rgb, cv2.COLOR_RGB2BGR)
        cv_gray = cv2.cvtColor(hwc_bgr, cv2.COLOR_BGR2GRAY)

        max_diff = np.max(np.abs(
            onnx_gray_uint8.astype(np.int16) - cv_gray.astype(np.int16)
        ))
        assert max_diff <= 1, f"最大差: {max_diff}"
