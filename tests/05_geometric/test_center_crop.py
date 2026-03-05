"""センタークロップ (center_crop) モデルのテスト."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_DIR = PROJECT_ROOT / "models" / CATEGORY


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    if not (MODEL_DIR / "center_crop.onnx").exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_DIR / "center_crop.onnx"))


def _run(session, img: np.ndarray, ratio: float = 0.5) -> np.ndarray:
    return session.run(None, {
        "input": img,
        "crop_ratio": np.array([ratio], dtype=np.float32),
    })[0]


class TestCenterCropOutputShape:
    def test_ratio_1_same_size(self, session):
        """ratio=1.0 で入力と同一サイズ."""
        img = np.random.rand(1, 3, 16, 20).astype(np.float32)
        out = _run(session, img, ratio=1.0)
        assert out.shape == (1, 3, 16, 20)

    def test_ratio_half(self, session):
        """ratio=0.5 で約半分のサイズ."""
        img = np.random.rand(1, 3, 16, 20).astype(np.float32)
        out = _run(session, img, ratio=0.5)
        assert out.shape == (1, 3, 8, 10)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 16, 20).astype(np.float32)
        out = _run(session, img, ratio=0.5)
        assert out.shape[0] == 2
        assert out.shape[1] == 3


class TestCenterCropValues:
    def test_ratio_1_identity(self, session):
        """ratio=1.0 で入力と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 20), dtype=np.float32)
        out = _run(session, img, ratio=1.0)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_center_region(self, session):
        """中央部分が正しく切り出される."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 20), dtype=np.float32)
        out = _run(session, img, ratio=0.5)
        # crop_h = floor(16*0.5)=8, crop_w = floor(20*0.5)=10
        # off_h = floor((16-8)/2)=4, off_w = floor((20-10)/2)=5
        expected = img[:, :, 4:12, 5:15]
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_uniform_image(self, session):
        """均一画像は切り出し後も同一値."""
        img = np.full((1, 3, 16, 16), 0.7, dtype=np.float32)
        out = _run(session, img, ratio=0.5)
        np.testing.assert_allclose(out, 0.7, atol=1e-5)

    def test_quarter_crop(self, session):
        """ratio=0.25 で 1/4 の中央を切り出す."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 20, 20), dtype=np.float32)
        out = _run(session, img, ratio=0.25)
        # crop = floor(20*0.25)=5, off = floor((20-5)/2)=7
        expected = img[:, :, 7:12, 7:12]
        np.testing.assert_allclose(out, expected, atol=1e-5)
