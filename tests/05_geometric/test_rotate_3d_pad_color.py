"""3D 回転 (rotate_3d_pad_color) モデルのテスト.

RGB パディング色指定付きの rotate_3d_pad_color モデルを検証する.
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

_FOCAL_LENGTH = 2.0


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    if not (MODEL_DIR / "rotate_3d_pad_color.onnx").exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_DIR / "rotate_3d_pad_color.onnx"))


def _run(session, img: np.ndarray,
         angle_x=0.0, angle_y=0.0, angle_z=0.0, zoom=1.0,
         pad_r=0.0, pad_g=0.0, pad_b=0.0) -> np.ndarray:
    return session.run(None, {
        "input": img,
        "angle_x": np.array([angle_x], dtype=np.float32),
        "angle_y": np.array([angle_y], dtype=np.float32),
        "angle_z": np.array([angle_z], dtype=np.float32),
        "zoom": np.array([zoom], dtype=np.float32),
        "pad_r": np.array([pad_r], dtype=np.float32),
        "pad_g": np.array([pad_g], dtype=np.float32),
        "pad_b": np.array([pad_b], dtype=np.float32),
    })[0]


class TestRotate3dPadColorOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        img = np.random.rand(1, 3, 16, 20).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 16, 20)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 12).astype(np.float32)
        out = _run(session, img, angle_x=10.0, angle_y=20.0, angle_z=30.0)
        assert out.shape == (2, 3, 8, 12)

    def test_dynamic_hw(self, session):
        img = np.random.rand(1, 3, 32, 48).astype(np.float32)
        out = _run(session, img, angle_z=45.0)
        assert out.shape == (1, 3, 32, 48)


class TestRotate3dPadColorValues:
    """出力値の正確性を検証するテスト群."""

    def test_identity(self, session):
        """全角度0・zoom=1 で入力と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 20), dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_uniform_image(self, session):
        """均一画像は回転後も変化しないこと (zoom=3 で全域がソース内)."""
        img = np.full((1, 3, 16, 16), 0.5, dtype=np.float32)
        out = _run(session, img, angle_x=10.0, angle_y=10.0, angle_z=10.0,
                   zoom=3.0)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_no_nan_inf(self, session):
        """極端な角度でも NaN/Inf が発生しないこと."""
        rng = np.random.default_rng(99)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, angle_x=90.0, angle_y=90.0, angle_z=90.0,
                   zoom=2.0)
        assert np.isfinite(out).all()

    def test_output_range(self, session):
        """出力が [0, 1] 範囲に収まること."""
        rng = np.random.default_rng(55)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img, angle_x=45.0, angle_y=-30.0, angle_z=60.0,
                   zoom=1.5, pad_r=0.5, pad_g=0.3, pad_b=0.8)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_zoom_effect(self, session):
        """zoom > 1 で中心付近が拡大されること."""
        rng = np.random.default_rng(33)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out_z1 = _run(session, img, zoom=1.0)
        out_z2 = _run(session, img, zoom=2.0)
        assert not np.allclose(out_z1, out_z2, atol=1e-3)

    def test_padding_color_black(self, session):
        """pad=(0,0,0) で範囲外が黒になること."""
        img = np.full((1, 3, 8, 8), 0.8, dtype=np.float32)
        out = _run(session, img, angle_y=60.0, pad_r=0.0, pad_g=0.0, pad_b=0.0)
        assert out.min() < 0.01

    def test_padding_color_white(self, session):
        """pad=(1,1,1) で範囲外が白になること."""
        img = np.full((1, 3, 8, 8), 0.2, dtype=np.float32)
        out = _run(session, img, angle_y=60.0, pad_r=1.0, pad_g=1.0, pad_b=1.0)
        assert out.max() > 0.99

    def test_padding_color_rgb(self, session):
        """パディング色が各チャネルで正しく適用されること."""
        img = np.zeros((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, angle_y=80.0,
                   pad_r=1.0, pad_g=0.5, pad_b=0.25)
        r_max = out[0, 0].max()
        g_max = out[0, 1].max()
        b_max = out[0, 2].max()
        assert r_max > g_max > b_max


class TestRotate3dPadColorVsOpenCV:
    """OpenCV warpPerspective との比較テスト群."""

    def _build_homography(self, H, W, angle_x, angle_y, angle_z, zoom):
        """中心基準 3D 回転のホモグラフィ行列を構築する."""
        ax = np.radians(angle_x)
        ay = np.radians(angle_y)
        az = np.radians(angle_z)
        f = _FOCAL_LENGTH

        cx, sx = np.cos(ax), np.sin(ax)
        cy, sy = np.cos(ay), np.sin(ay)
        cz, sz = np.cos(az), np.sin(az)

        R = np.array([
            [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
            [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
            [-sy,   cy*sx,            cy*cx],
        ])
        Rt = R.T

        rt00, rt01, rt02 = Rt[0]
        rt10, rt11, rt12 = Rt[1]
        rt20, rt21, rt22 = Rt[2]

        H_norm = np.array([
            [f * (rt22 * rt00 - rt02 * rt20) / zoom,
             f * (rt22 * rt01 - rt02 * rt21) / zoom,
             f * f * (rt22 * rt02 - rt02 * rt22)],
            [f * (rt22 * rt10 - rt12 * rt20) / zoom,
             f * (rt22 * rt11 - rt12 * rt21) / zoom,
             f * f * (rt22 * rt12 - rt12 * rt22)],
            [rt20 / zoom,
             rt21 / zoom,
             rt22 * f],
        ])

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
        M_pix = P_inv @ H_norm @ P
        return M_pix.astype(np.float64)

    def test_matches_opencv_identity(self, session):
        """identity 変換で OpenCV warpPerspective と一致すること."""
        H, W = 16, 20
        rng = np.random.default_rng(42)
        img_nchw = rng.random((1, 3, H, W), dtype=np.float32)

        onnx_out = _run(session, img_nchw)
        onnx_u8 = (onnx_out[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)

        hwc_u8 = (img_nchw[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        M = self._build_homography(H, W, 0, 0, 0, 1.0)
        cv_out = cv2.warpPerspective(
            hwc_u8, M, (W, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )

        diff = np.abs(onnx_u8.astype(np.int16) - cv_out.astype(np.int16))
        assert diff.max() <= 1, f"最大誤差 {diff.max()} > 1"

    def test_matches_opencv_z_rotation(self, session):
        """Z 軸回転で OpenCV warpPerspective と概ね一致すること."""
        H, W = 32, 32
        angle_z = 30.0

        rng = np.random.default_rng(123)
        img_nchw = rng.random((1, 3, H, W), dtype=np.float32)

        onnx_out = _run(session, img_nchw, angle_z=angle_z)
        onnx_u8 = (onnx_out[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)

        hwc_u8 = (img_nchw[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        M = self._build_homography(H, W, 0, 0, angle_z, 1.0)
        cv_out = cv2.warpPerspective(
            hwc_u8, M, (W, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )

        diff = np.abs(onnx_u8.astype(np.int16) - cv_out.astype(np.int16))
        assert diff.max() <= 5, f"最大誤差 {diff.max()} > 5"

    def test_matches_opencv_3d_rotation(self, session):
        """3D 回転で OpenCV warpPerspective と概ね一致すること."""
        H, W = 32, 32
        ax, ay, az, zoom = 15.0, 20.0, 10.0, 1.2

        rng = np.random.default_rng(456)
        img_nchw = rng.random((1, 3, H, W), dtype=np.float32)

        onnx_out = _run(session, img_nchw,
                        angle_x=ax, angle_y=ay, angle_z=az, zoom=zoom)
        onnx_u8 = (onnx_out[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)

        hwc_u8 = (img_nchw[0].transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        M = self._build_homography(H, W, ax, ay, az, zoom)
        cv_out = cv2.warpPerspective(
            hwc_u8, M, (W, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )

        diff = np.abs(onnx_u8.astype(np.int16) - cv_out.astype(np.int16))
        assert diff.max() <= 5, f"最大誤差 {diff.max()} > 5"
