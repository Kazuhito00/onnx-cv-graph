"""カラーマップ適用モデルのテスト.

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

LUMA_WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)

COLORMAP_NAMES = ["jet", "turbo", "inferno", "viridis"]
COLORMAP_IDS = {
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    for name in COLORMAP_NAMES:
        p = PROJECT_ROOT / "models" / CATEGORY / f"colormap_{name}.onnx"
        if not p.exists():
            subprocess.check_call(
                [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
                cwd=str(PROJECT_ROOT),
            )
            break


@pytest.fixture(scope="module", params=COLORMAP_NAMES)
def session_and_name(request):
    name = request.param
    p = PROJECT_ROOT / "models" / CATEGORY / f"colormap_{name}.onnx"
    return ort.InferenceSession(str(p)), name


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


def _numpy_colormap(img: np.ndarray, colormap_id: int) -> np.ndarray:
    """NumPy + OpenCV 参照実装."""
    # グレースケール化
    gray = (img * LUMA_WEIGHTS).sum(axis=1, keepdims=True)  # (N, 1, H, W)
    # 量子化
    indices = np.floor(gray * 255).clip(0, 255).astype(np.uint8)
    # LUT 構築
    gray_bar = np.arange(256, dtype=np.uint8).reshape(256, 1)
    bgr_lut = cv2.applyColorMap(gray_bar, colormap_id)
    rgb_lut = bgr_lut[:, 0, ::-1].astype(np.float32) / 255.0  # (256, 3)
    # ルックアップ
    N, _, H, W = img.shape
    flat_idx = indices.reshape(-1)
    mapped = rgb_lut[flat_idx]  # (N*H*W, 3)
    return mapped.reshape(N, H, W, 3).transpose(0, 3, 1, 2)  # (N, 3, H, W)


class TestColormapOutputShape:
    def test_single_image(self, session_and_name):
        sess, name = session_and_name
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (1, 3, 16, 16)

    def test_batch(self, session_and_name):
        sess, name = session_and_name
        img = np.random.rand(2, 3, 16, 16).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (2, 3, 16, 16)


class TestColormapValues:
    def test_output_range(self, session_and_name):
        """出力は [0, 1] の範囲内."""
        sess, name = session_and_name
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(sess, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_matches_opencv_reference(self, session_and_name):
        """OpenCV の applyColorMap と一致すること."""
        sess, name = session_and_name
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        expected = _numpy_colormap(img, COLORMAP_IDS[name])
        out = _run(sess, img)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_black_input(self, session_and_name):
        """全黒入力で LUT の index=0 の色が出ること."""
        sess, name = session_and_name
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        out = _run(sess, img)
        # 全ピクセル同一色であること
        assert np.allclose(out[:, :, 0, 0], out[:, :, 2, 2])

    def test_white_input(self, session_and_name):
        """全白入力で LUT の index=255 の色が出ること."""
        sess, name = session_and_name
        img = np.ones((1, 3, 4, 4), dtype=np.float32)
        out = _run(sess, img)
        assert np.allclose(out[:, :, 0, 0], out[:, :, 2, 2])

    def test_different_colormaps_differ(self):
        """異なるカラーマップで出力が異なること."""
        jet_path = PROJECT_ROOT / "models" / CATEGORY / "colormap_jet.onnx"
        turbo_path = PROJECT_ROOT / "models" / CATEGORY / "colormap_turbo.onnx"
        jet_sess = ort.InferenceSession(str(jet_path))
        hot_sess = ort.InferenceSession(str(turbo_path))
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        jet_out = _run(jet_sess, img)
        hot_out = _run(hot_sess, img)
        assert not np.array_equal(jet_out, hot_out)
