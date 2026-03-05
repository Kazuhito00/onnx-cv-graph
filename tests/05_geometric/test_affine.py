"""アフィン変換 (affine) モデルのテスト.

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
    if not (MODEL_DIR / "affine.onnx").exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_DIR / "affine.onnx"))


def _run(session, img: np.ndarray, a=1.0, b=0.0, tx=0.0,
         c=0.0, d=1.0, ty=0.0) -> np.ndarray:
    return session.run(None, {
        "input": img,
        "a": np.array([a], dtype=np.float32),
        "b": np.array([b], dtype=np.float32),
        "tx": np.array([tx], dtype=np.float32),
        "c": np.array([c], dtype=np.float32),
        "d": np.array([d], dtype=np.float32),
        "ty": np.array([ty], dtype=np.float32),
    })[0]


class TestAffineOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        img = np.random.rand(1, 3, 16, 20).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 16, 20)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 12).astype(np.float32)
        out = _run(session, img, a=0.5, d=0.5)
        assert out.shape == (2, 3, 8, 12)


class TestAffineValues:
    """出力値の正確性を検証するテスト群."""

    def test_identity(self, session):
        """identity 変換 (a=1,b=0,tx=0,c=0,d=1,ty=0) で入力と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 20), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_uniform_image_unchanged(self, session):
        """均一画像はアフィン変換後も変化しないこと (範囲外は zeros)."""
        img = np.full((1, 3, 16, 16), 0.5, dtype=np.float32)
        out = _run(session, img, a=0.5, d=0.5)
        # スケーリングで全ピクセルがソース画像内に収まる
        np.testing.assert_allclose(out, 0.5, atol=1e-5)


class TestAffineVsOpenCV:
    """OpenCV warpAffine との比較テスト群."""

    def _norm_to_pixel_matrix(self, H, W, a, b, tx, c, d, ty):
        """正規化座標空間のアフィンパラメータをピクセル空間の 2×3 行列に変換する."""
        # 正規化→ピクセル: P_inv = [[(W-1)/2, 0, (W-1)/2], [0, (H-1)/2, (H-1)/2], [0, 0, 1]]
        # ピクセル→正規化: P = [[2/(W-1), 0, -1], [0, 2/(H-1), -1], [0, 0, 1]]
        # ピクセル空間行列: M_pix = P_inv @ M_norm @ P
        P = np.array([
            [2.0 / (W - 1), 0, -1],
            [0, 2.0 / (H - 1), -1],
            [0, 0, 1],
        ])
        P_inv = np.array([
            [(W - 1) / 2.0, 0, (W - 1) / 2.0],
            [0, (H - 1) / 2.0, (H - 1) / 2.0],
            [0, 0, 1],
        ])
        M_norm = np.array([
            [a, b, tx],
            [c, d, ty],
            [0, 0, 1],
        ])
        M_pix = P_inv @ M_norm @ P
        return M_pix[:2].astype(np.float32)

    def test_matches_opencv_identity(self, session):
        """identity 変換で OpenCV warpAffine と一致すること."""
        H, W = 16, 20
        rng = np.random.default_rng(42)
        img_nchw = rng.random((1, 3, H, W), dtype=np.float32)

        onnx_out = _run(session, img_nchw)
        onnx_u8 = (onnx_out[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)

        hwc_u8 = (img_nchw[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        M = self._norm_to_pixel_matrix(H, W, 1, 0, 0, 0, 1, 0)
        cv_out = cv2.warpAffine(
            hwc_u8, M, (W, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )

        diff = np.abs(onnx_u8.astype(np.int16) - cv_out.astype(np.int16))
        assert diff.max() <= 1, f"最大誤差 {diff.max()} > 1"

    def test_matches_opencv_scale(self, session):
        """スケーリング変換で OpenCV warpAffine と一致すること."""
        H, W = 16, 16
        a, b, tx = 0.5, 0.0, 0.0
        c, d, ty = 0.0, 0.5, 0.0

        rng = np.random.default_rng(99)
        img_nchw = rng.random((1, 3, H, W), dtype=np.float32)

        onnx_out = _run(session, img_nchw, a=a, b=b, tx=tx, c=c, d=d, ty=ty)
        onnx_u8 = (onnx_out[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)

        hwc_u8 = (img_nchw[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        M = self._norm_to_pixel_matrix(H, W, a, b, tx, c, d, ty)
        cv_out = cv2.warpAffine(
            hwc_u8, M, (W, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )

        diff = np.abs(onnx_u8.astype(np.int16) - cv_out.astype(np.int16))
        assert diff.max() <= 2, f"最大誤差 {diff.max()} > 2"
