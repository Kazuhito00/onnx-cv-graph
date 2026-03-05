"""射影変換 (perspective) モデルのテスト.

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
    if not (MODEL_DIR / "perspective.onnx").exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_DIR / "perspective.onnx"))


def _run(session, img: np.ndarray,
         p00=1.0, p01=0.0, p02=0.0,
         p10=0.0, p11=1.0, p12=0.0,
         p20=0.0, p21=0.0) -> np.ndarray:
    return session.run(None, {
        "input": img,
        "p00": np.array([p00], dtype=np.float32),
        "p01": np.array([p01], dtype=np.float32),
        "p02": np.array([p02], dtype=np.float32),
        "p10": np.array([p10], dtype=np.float32),
        "p11": np.array([p11], dtype=np.float32),
        "p12": np.array([p12], dtype=np.float32),
        "p20": np.array([p20], dtype=np.float32),
        "p21": np.array([p21], dtype=np.float32),
    })[0]


class TestPerspectiveOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        img = np.random.rand(1, 3, 16, 20).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 16, 20)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 12).astype(np.float32)
        out = _run(session, img, p00=0.8, p11=0.8)
        assert out.shape == (2, 3, 8, 12)


class TestPerspectiveValues:
    """出力値の正確性を検証するテスト群."""

    def test_identity(self, session):
        """identity 変換で入力と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 20), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_affine_subset(self, session):
        """p20=p21=0 のときアフィン変換と一致すること."""
        rng = np.random.default_rng(77)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)

        # p20=p21=0 → アフィン部分のみ (スケーリング 0.5)
        out_persp = _run(session, img, p00=0.5, p11=0.5, p20=0.0, p21=0.0)

        # AffineOp で同じ変換
        affine_sess = ort.InferenceSession(str(MODEL_DIR / "affine.onnx"))
        out_affine = affine_sess.run(None, {
            "input": img,
            "a": np.array([0.5], dtype=np.float32),
            "b": np.array([0.0], dtype=np.float32),
            "tx": np.array([0.0], dtype=np.float32),
            "c": np.array([0.0], dtype=np.float32),
            "d": np.array([0.5], dtype=np.float32),
            "ty": np.array([0.0], dtype=np.float32),
        })[0]

        np.testing.assert_allclose(out_persp, out_affine, atol=1e-5)

    def test_uniform_image(self, session):
        """均一画像は射影変換後も変化しないこと (スケール 0.5 で全域がソース内)."""
        img = np.full((1, 3, 16, 16), 0.5, dtype=np.float32)
        out = _run(session, img, p00=0.5, p11=0.5)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)


class TestPerspectiveVsOpenCV:
    """OpenCV warpPerspective との比較テスト群."""

    def _norm_to_pixel_matrix(self, H, W, p00, p01, p02, p10, p11, p12, p20, p21):
        """正規化座標空間のホモグラフィパラメータをピクセル空間の 3×3 行列に変換する."""
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
            [p00, p01, p02],
            [p10, p11, p12],
            [p20, p21, 1],
        ])
        M_pix = P_inv @ M_norm @ P
        return M_pix.astype(np.float64)

    def test_matches_opencv_identity(self, session):
        """identity 変換で OpenCV warpPerspective と一致すること."""
        H, W = 16, 20
        rng = np.random.default_rng(42)
        img_nchw = rng.random((1, 3, H, W), dtype=np.float32)

        onnx_out = _run(session, img_nchw)
        onnx_u8 = (onnx_out[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)

        hwc_u8 = (img_nchw[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        M = self._norm_to_pixel_matrix(H, W, 1, 0, 0, 0, 1, 0, 0, 0)
        cv_out = cv2.warpPerspective(
            hwc_u8, M, (W, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )

        diff = np.abs(onnx_u8.astype(np.int16) - cv_out.astype(np.int16))
        assert diff.max() <= 1, f"最大誤差 {diff.max()} > 1"

    def test_matches_opencv_perspective(self, session):
        """射影変換で OpenCV warpPerspective と概ね一致すること."""
        H, W = 32, 32
        p00, p01, p02 = 0.9, 0.1, 0.0
        p10, p11, p12 = -0.1, 0.9, 0.0
        p20, p21 = 0.05, 0.03

        rng = np.random.default_rng(123)
        img_nchw = rng.random((1, 3, H, W), dtype=np.float32)

        onnx_out = _run(session, img_nchw,
                        p00=p00, p01=p01, p02=p02,
                        p10=p10, p11=p11, p12=p12,
                        p20=p20, p21=p21)
        onnx_u8 = (onnx_out[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)

        hwc_u8 = (img_nchw[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        M = self._norm_to_pixel_matrix(H, W, p00, p01, p02, p10, p11, p12, p20, p21)
        cv_out = cv2.warpPerspective(
            hwc_u8, M, (W, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )

        diff = np.abs(onnx_u8.astype(np.int16) - cv_out.astype(np.int16))
        # 射影変換は非線形な座標マッピングのため float32 精度差が大きくなりやすい
        assert diff.max() <= 5, f"最大誤差 {diff.max()} > 5"
