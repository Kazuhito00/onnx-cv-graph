"""閾値2値化モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "binarize.onnx"

LUMA_WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)


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


def _run(session, img: np.ndarray, threshold: float) -> np.ndarray:
    """セッションで推論を実行し、最初の出力を返すヘルパー."""
    thr = np.array([threshold], dtype=np.float32)
    return session.run(None, {"input": img, "threshold": thr})[0]


def _numpy_binarize(img: np.ndarray, threshold: float) -> np.ndarray:
    """NumPy 参照実装: グレースケール化して閾値で2値化し、3ch に拡張."""
    gray = (img * LUMA_WEIGHTS).sum(axis=1, keepdims=True)
    bin_1ch = (gray > threshold).astype(np.float32)
    return np.repeat(bin_1ch, 3, axis=1)


class TestBinarizeOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        """単一画像 (N=1) で出力が (1,3,H,W) になること."""
        img = np.random.rand(1, 3, 4, 4).astype(np.float32)
        out = _run(session, img, 0.5)
        assert out.shape == (1, 3, 4, 4)

    def test_batch(self, session):
        """バッチ (N>1) で出力が (N,3,H,W) になること."""
        img = np.random.rand(3, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.5)
        assert out.shape == (3, 3, 8, 8)

    def test_all_channels_equal(self, session):
        """出力の3チャネルが全て同一値であること."""
        rng = np.random.default_rng(99)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 0.5)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        np.testing.assert_array_equal(out[:, 0], out[:, 2])


class TestBinarizeValues:
    """出力値の正確性を検証するテスト群."""

    def test_output_only_zero_or_one(self, session):
        """出力値が 0.0 と 1.0 のみであること."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        out = _run(session, img, 0.5)
        unique = np.unique(out)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_threshold_behavior(self, session):
        """threshold=0.5 で gray>0.5 → 1.0, gray≤0.5 → 0.0 になること."""
        # gray が約 0.7 になる画素 (全チャネル 0.7)
        img_bright = np.full((1, 3, 1, 1), 0.7, dtype=np.float32)
        out_bright = _run(session, img_bright, 0.5)
        assert out_bright[0, 0, 0, 0] == 1.0

        # gray が約 0.3 になる画素 (全チャネル 0.3)
        img_dark = np.full((1, 3, 1, 1), 0.3, dtype=np.float32)
        out_dark = _run(session, img_dark, 0.5)
        assert out_dark[0, 0, 0, 0] == 0.0

    def test_all_white(self, session):
        """全白 (1,1,1) → 全 1.0 になること."""
        img = np.ones((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img, 0.5)
        np.testing.assert_array_equal(out, np.ones((1, 3, 4, 4), dtype=np.float32))

    def test_all_black(self, session):
        """全黒 (0,0,0) → 全 0.0 になること."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img, 0.5)
        np.testing.assert_array_equal(out, np.zeros((1, 3, 4, 4), dtype=np.float32))

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        threshold = 0.4
        expected = _numpy_binarize(img, threshold)
        out = _run(session, img, threshold)
        np.testing.assert_array_equal(out, expected)


def _nchw_to_hwc_uint8(nchw: np.ndarray) -> np.ndarray:
    """NCHW float32 [0,1] → HWC uint8 [0,255] に変換するヘルパー (単一画像)."""
    return (nchw[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)


class TestBinarizeVsOpenCV:
    """OpenCV の threshold(THRESH_BINARY) との比較テスト群.

    OpenCV は uint8 で動作するため量子化誤差が生じる.
    ONNX 出力を uint8 に変換した上で許容誤差 1 以内で比較する.
    """

    def test_random_image(self, session):
        """ランダム画像で OpenCV threshold との一致を検証.

        ONNX は float32 グレースケール値で比較し、OpenCV は uint8 量子化後に
        比較するため、閾値境界付近の画素では判定が異なりうる.
        境界から十分離れた画素のみで完全一致を確認する.
        """
        rng = np.random.default_rng(123)
        img_nchw = rng.random((1, 3, 32, 32), dtype=np.float32)
        threshold = 0.5

        # ONNX 推論 → (1,3,H,W) から ch0 を取得 (3ch同一値)
        onnx_out = _run(session, img_nchw, threshold)
        onnx_uint8 = (onnx_out[0, 0] * 255.0).clip(0, 255).round().astype(np.uint8)

        # OpenCV: RGB→BGR→グレースケール→閾値2値化
        hwc_rgb = _nchw_to_hwc_uint8(img_nchw)
        hwc_bgr = cv2.cvtColor(hwc_rgb, cv2.COLOR_RGB2BGR)
        cv_gray = cv2.cvtColor(hwc_bgr, cv2.COLOR_BGR2GRAY)
        thr_uint8 = int(threshold * 255)
        _, cv_bin = cv2.threshold(cv_gray, thr_uint8, 255, cv2.THRESH_BINARY)

        # 閾値境界付近 (±2) の画素を除外して比較
        # uint8 量子化による判定差はこの範囲内でのみ発生する
        non_boundary = np.abs(cv_gray.astype(np.int16) - thr_uint8) > 2
        assert non_boundary.sum() > 0, "境界外の画素が存在すること"
        np.testing.assert_array_equal(
            onnx_uint8[non_boundary],
            cv_bin[non_boundary],
        )
