"""反転 (flip) モデルのテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
HFlip / VFlip / HVFlip の全バリアントをテストする.
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

MODEL_NAMES = ["hflip", "vflip", "hvflip"]


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    missing = any(not (MODEL_DIR / f"{n}.onnx").exists() for n in MODEL_NAMES)
    if missing:
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module", params=MODEL_NAMES, ids=MODEL_NAMES)
def session_and_name(request):
    """(session, model_name) のタプルを返す."""
    name = request.param
    sess = ort.InferenceSession(str(MODEL_DIR / f"{name}.onnx"))
    return sess, name


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


class TestFlipOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session_and_name):
        sess, _ = session_and_name
        img = np.random.rand(1, 3, 8, 12).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (1, 3, 8, 12)

    def test_batch(self, session_and_name):
        sess, _ = session_and_name
        img = np.random.rand(2, 3, 8, 12).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (2, 3, 8, 12)


class TestFlipValues:
    """出力値の正確性を検証するテスト群."""

    def test_double_flip_identity(self, session_and_name):
        """同じ反転を2回適用すると元に戻ること."""
        sess, _ = session_and_name
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 12), dtype=np.float32)
        out1 = _run(sess, img)
        out2 = _run(sess, out1)
        np.testing.assert_allclose(out2, img, atol=1e-5)

    def test_uniform_image_unchanged(self, session_and_name):
        """均一画像は反転後も変化しないこと."""
        sess, _ = session_and_name
        img = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_allclose(out, img, atol=1e-5)


class TestFlipVsOpenCV:
    """OpenCV flip との比較テスト群."""

    def test_matches_opencv(self, session_and_name):
        """ランダム画像で OpenCV flip との一致を検証."""
        sess, name = session_and_name
        rng = np.random.default_rng(123)
        img_nchw = rng.random((1, 3, 16, 20), dtype=np.float32)

        onnx_out = _run(sess, img_nchw)
        onnx_uint8 = (onnx_out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)

        hwc_u8 = (img_nchw[0].transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)
        flip_code = {"hflip": 1, "vflip": 0, "hvflip": -1}[name]
        cv_out = cv2.flip(hwc_u8, flip_code)

        diff = np.abs(onnx_uint8.astype(np.int16) - cv_out.astype(np.int16))
        assert diff.max() <= 1, f"最大誤差 {diff.max()} > 1 ({name})"
