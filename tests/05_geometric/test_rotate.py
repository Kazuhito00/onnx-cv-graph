"""回転 (rotation) モデルのテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
90° / 180° / 270° の全バリアントをテストする.
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

ANGLES = [90, 180, 270]


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    missing = any(
        not (MODEL_DIR / f"rotate_{a}.onnx").exists() for a in ANGLES
    )
    if missing:
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module", params=ANGLES, ids=[f"rotate_{a}" for a in ANGLES])
def session_and_angle(request):
    """(session, angle) のタプルを返す."""
    a = request.param
    sess = ort.InferenceSession(str(MODEL_DIR / f"rotate_{a}.onnx"))
    return sess, a


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


class TestRotateOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session_and_angle):
        sess, a = session_and_angle
        img = np.random.rand(1, 3, 8, 12).astype(np.float32)
        out = _run(sess, img)
        if a == 180:
            assert out.shape == (1, 3, 8, 12)
        else:
            # 90° / 270° は H と W が入れ替わる
            assert out.shape == (1, 3, 12, 8)

    def test_batch(self, session_and_angle):
        sess, a = session_and_angle
        img = np.random.rand(2, 3, 8, 12).astype(np.float32)
        out = _run(sess, img)
        assert out.shape[0] == 2
        assert out.shape[1] == 3


class TestRotateValues:
    """出力値の正確性を検証するテスト群."""

    def test_four_rotations_identity(self):
        """90° を4回適用すると元に戻ること."""
        sess = ort.InferenceSession(str(MODEL_DIR / "rotate_90.onnx"))
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = img
        for _ in range(4):
            out = _run(sess, out)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_uniform_image_unchanged(self, session_and_angle):
        """均一画像は回転後も変化しないこと."""
        sess, _ = session_and_angle
        img = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_180_is_double_90(self):
        """180° 回転は 90° を2回適用した結果と一致すること."""
        sess90 = ort.InferenceSession(str(MODEL_DIR / "rotate_90.onnx"))
        sess180 = ort.InferenceSession(str(MODEL_DIR / "rotate_180.onnx"))
        rng = np.random.default_rng(99)
        img = rng.random((1, 3, 8, 12), dtype=np.float32)
        out_2x90 = _run(sess90, _run(sess90, img))
        out_180 = _run(sess180, img)
        np.testing.assert_allclose(out_180, out_2x90, atol=1e-5)


class TestRotateVsOpenCV:
    """OpenCV rotate との比較テスト群."""

    def test_matches_opencv(self, session_and_angle):
        """ランダム画像で OpenCV rotate との一致を検証."""
        sess, a = session_and_angle
        rng = np.random.default_rng(123)
        img_nchw = rng.random((1, 3, 16, 20), dtype=np.float32)

        onnx_out = _run(sess, img_nchw)
        onnx_uint8 = (onnx_out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)

        hwc_u8 = (img_nchw[0].transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)
        rotate_code = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }[a]
        cv_out = cv2.rotate(hwc_u8, rotate_code)

        diff = np.abs(onnx_uint8.astype(np.int16) - cv_out.astype(np.int16))
        assert diff.max() <= 1, f"最大誤差 {diff.max()} > 1 (rotate_{a})"
