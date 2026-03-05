"""HSV 範囲抽出モデルのテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "hsv_range.onnx"


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    if not MODEL_PATH.exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )
    assert MODEL_PATH.exists()


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_PATH))


def _run(
    session, img: np.ndarray,
    h_min: float, h_max: float,
    s_min: float, s_max: float,
    v_min: float, v_max: float,
) -> np.ndarray:
    return session.run(None, {
        "input": img,
        "h_min": np.array([h_min], dtype=np.float32),
        "h_max": np.array([h_max], dtype=np.float32),
        "s_min": np.array([s_min], dtype=np.float32),
        "s_max": np.array([s_max], dtype=np.float32),
        "v_min": np.array([v_min], dtype=np.float32),
        "v_max": np.array([v_max], dtype=np.float32),
    })[0]


def _rgb_to_hsv_numpy(rgb: np.ndarray) -> np.ndarray:
    """NumPy による RGB→HSV 変換 (OpenCV 準拠, [0,1] 正規化).

    入力: (N, 3, H, W) float32 [0,1]
    出力: (N, 3, H, W) float32 H,S,V 各 [0,1]
    """
    R = rgb[:, 0:1]
    G = rgb[:, 1:2]
    B = rgb[:, 2:3]
    V = np.maximum(np.maximum(R, G), B)
    Vmin = np.minimum(np.minimum(R, G), B)
    diff = V - Vmin

    # S
    S = np.where(V > 0, diff / (V + 1e-7), 0.0)

    # H
    diff_safe = diff + 1e-7
    h_r = (G - B) / diff_safe
    h_g = 2.0 + (B - R) / diff_safe
    h_b = 4.0 + (R - G) / diff_safe

    H = np.where(V == B, h_b, h_r)
    H = np.where(V == G, h_g, H)
    H = H * 60.0
    H = np.where(H < 0, H + 360.0, H)
    H = H / 360.0
    H = np.where(diff == 0, 0.0, H)

    return np.concatenate([H, S, V], axis=1).astype(np.float32)


class TestHsvRangeOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        assert out.shape == (2, 3, 8, 8)


class TestHsvRangeValues:
    def test_full_range_all_ones(self, session):
        """全範囲指定で全画素 1.0."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_empty_range_all_zeros(self, session):
        """S 範囲を不可能に設定で全画素 0.0."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        # s_min=0.9, s_max=0.91 → ほぼ全画素が範囲外
        # V 範囲を不可能にする方が確実
        out = _run(session, img, 0.0, 1.0, 0.0, 1.0, 0.99, 0.991)
        # ランダム画像で V が [0.99, 0.991] の範囲内に入るのはほぼゼロ
        assert out.sum() < img.size * 0.01

    def test_output_is_binary(self, session):
        """出力は 0.0 か 1.0 のみ."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 0.1, 0.5, 0.2, 0.8, 0.3, 0.9)
        unique = np.unique(out)
        for v in unique:
            assert v == pytest.approx(0.0, abs=1e-6) or v == pytest.approx(1.0, abs=1e-6)

    def test_3ch_identical(self, session):
        """出力3チャネルが全て同一."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 0.1, 0.5, 0.2, 0.8, 0.3, 0.9)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        np.testing.assert_array_equal(out[:, 0], out[:, 2])

    def test_pure_red_detected(self, session):
        """純赤 (H≈0) が赤範囲で検出される."""
        # 純赤: R=1, G=0, B=0 → H=0, S=1, V=1
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        img[:, 0] = 1.0  # R=1
        # 赤の H 範囲: 0~0.05 (0°~18°)
        out = _run(session, img, 0.0, 0.05, 0.5, 1.0, 0.5, 1.0)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_pure_green_detected(self, session):
        """純緑 (H≈0.333) が緑範囲で検出される."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        img[:, 1] = 1.0  # G=1 → H=120°=0.333
        out = _run(session, img, 0.28, 0.39, 0.5, 1.0, 0.5, 1.0)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_pure_blue_detected(self, session):
        """純青 (H≈0.667) が青範囲で検出される."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        img[:, 2] = 1.0  # B=1 → H=240°=0.667
        out = _run(session, img, 0.6, 0.72, 0.5, 1.0, 0.5, 1.0)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_pure_green_not_in_red_range(self, session):
        """純緑は赤範囲で検出されない."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        img[:, 1] = 1.0  # G=1
        out = _run(session, img, 0.0, 0.05, 0.5, 1.0, 0.5, 1.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_hue_wrap_around(self, session):
        """色相の折り返し: h_min > h_max で赤領域を検出."""
        # 純赤: H=0
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        img[:, 0] = 1.0  # R=1 → H=0
        # h_min=0.9, h_max=0.1 → H≥0.9 OR H≤0.1 (赤の折り返し)
        out = _run(session, img, 0.9, 0.1, 0.5, 1.0, 0.5, 1.0)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_hue_wrap_excludes_green(self, session):
        """色相折り返しで緑は除外."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        img[:, 1] = 1.0  # G=1 → H=0.333
        # 折り返し赤範囲: H≥0.9 OR H≤0.1
        out = _run(session, img, 0.9, 0.1, 0.0, 1.0, 0.0, 1.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_gray_pixel_zero_saturation(self, session):
        """グレー画素 (S=0) は S>0 フィルタで除外."""
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)  # S=0, V=0.5
        out = _run(session, img, 0.0, 1.0, 0.1, 1.0, 0.0, 1.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_matches_numpy_reference(self, session):
        """NumPy 参照実装と一致."""
        rng = np.random.default_rng(123)
        img = rng.random((2, 3, 8, 8), dtype=np.float32)
        h_min, h_max = 0.15, 0.45
        s_min, s_max = 0.2, 0.9
        v_min, v_max = 0.1, 0.95

        out = _run(session, img, h_min, h_max, s_min, s_max, v_min, v_max)

        # NumPy 参照
        hsv = _rgb_to_hsv_numpy(img)
        H, S, V = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
        h_mask = (H >= h_min) & (H <= h_max)
        s_mask = (S >= s_min) & (S <= s_max)
        v_mask = (V >= v_min) & (V <= v_max)
        expected = (h_mask & s_mask & v_mask).astype(np.float32)
        expected = np.broadcast_to(expected, img.shape)

        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_v_filter_only(self, session):
        """V のみでフィルタリング (H, S は全範囲)."""
        img = np.zeros((1, 3, 2, 2), dtype=np.float32)
        # 画素0: 暗い (V=0.2)
        img[0, :, 0, 0] = 0.2
        # 画素1: 明るい (V=0.8)
        img[0, :, 0, 1] = 0.8
        # 画素2: 中間 (V=0.5)
        img[0, :, 1, 0] = 0.5
        # 画素3: 最大 (V=1.0)
        img[0, :, 1, 1] = 1.0

        # V >= 0.5 のみ抽出
        out = _run(session, img, 0.0, 1.0, 0.0, 1.0, 0.5, 1.0)
        assert out[0, 0, 0, 0] == pytest.approx(0.0, abs=1e-6)  # V=0.2 → 除外
        assert out[0, 0, 0, 1] == pytest.approx(1.0, abs=1e-6)  # V=0.8 → 含む
        assert out[0, 0, 1, 0] == pytest.approx(1.0, abs=1e-6)  # V=0.5 → 含む
        assert out[0, 0, 1, 1] == pytest.approx(1.0, abs=1e-6)  # V=1.0 → 含む
