"""リサイズモデルのテスト.

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
MODEL_DIR = PROJECT_ROOT / "models" / CATEGORY


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    if not (MODEL_DIR / "resize.onnx").exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_DIR / "resize.onnx"))


def _run(session, img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return session.run(None, {
        "input": img,
        "scale": np.array([scale], dtype=np.float32),
    })[0]


class TestResizeOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_scale_1(self, session):
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        out = _run(session, img, 1.0)
        assert out.shape == (1, 3, 16, 16)

    def test_scale_2(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 2.0)
        assert out.shape == (1, 3, 16, 16)

    def test_scale_half(self, session):
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        out = _run(session, img, 0.5)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 1.5)
        assert out.shape[0] == 2
        assert out.shape[1] == 3


class TestResizeValues:
    """出力値の正確性を検証するテスト群."""

    def test_scale_1_identity(self, session):
        """scale=1.0 のとき入力と一致すること."""
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        out = _run(session, img, 1.0)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_uniform_image(self, session):
        """均一画像はリサイズ後も同一値であること."""
        img = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)
        out = _run(session, img, 2.0)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)


class TestResizeVsOpenCV:
    """OpenCV resize との比較テスト群."""

    def test_matches_opencv_scale2(self, session):
        """scale=2 で OpenCV bilinear resize との一致を検証."""
        rng = np.random.default_rng(42)
        img_nchw = rng.random((1, 3, 16, 16), dtype=np.float32)

        onnx_out = _run(session, img_nchw, 2.0)
        onnx_uint8 = (onnx_out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)

        hwc_u8 = (img_nchw[0].transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)
        cv_out = cv2.resize(hwc_u8, (32, 32), interpolation=cv2.INTER_LINEAR)

        diff = np.abs(onnx_uint8.astype(np.int16) - cv_out.astype(np.int16))
        assert diff.max() <= 2, f"最大誤差 {diff.max()} > 2"
