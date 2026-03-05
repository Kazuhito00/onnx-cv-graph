"""HSV ベース色抑制モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "color_suppress.onnx"


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


def _run(session, img, h_center=0.0, h_range=0.08, s_min=0.3, strength=1.0):
    return session.run(None, {
        "input": img,
        "h_center": np.array([h_center], dtype=np.float32),
        "h_range": np.array([h_range], dtype=np.float32),
        "s_min": np.array([s_min], dtype=np.float32),
        "strength": np.array([strength], dtype=np.float32),
    })[0]


def _make_color_pixel(r, g, b):
    """単一ピクセルの (1, 3, 1, 1) テンソルを作成."""
    return np.array([[[[r]], [[g]], [[b]]]], dtype=np.float32)


class TestColorSuppressOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        assert _run(session, img).shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        assert _run(session, img).shape == (2, 3, 8, 8)


class TestColorSuppressValues:
    def test_strength_zero_unchanged(self, session):
        """strength=0 で入力と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, strength=0.0)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_red_stamp_removed(self, session):
        """純赤ピクセルが白に置換される."""
        img = _make_color_pixel(0.9, 0.1, 0.1)
        out = _run(session, img, h_center=0.0, h_range=0.1, s_min=0.3, strength=1.0)
        # 赤は高彩度 → マスクにマッチ → 白に
        np.testing.assert_allclose(out, 1.0, atol=0.05)

    def test_black_text_preserved(self, session):
        """黒テキスト (低彩度) は保持される."""
        img = _make_color_pixel(0.1, 0.1, 0.1)
        out = _run(session, img, h_center=0.0, h_range=0.5, s_min=0.3, strength=1.0)
        # 黒は彩度≈0 → s_min=0.3 で除外 → 変化なし
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_gray_text_preserved(self, session):
        """グレーテキスト (無彩色) は保持される."""
        img = _make_color_pixel(0.5, 0.5, 0.5)
        out = _run(session, img, h_center=0.0, h_range=0.5, s_min=0.3, strength=1.0)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_blue_not_affected_by_red_suppress(self, session):
        """赤抑制で青ピクセルは影響を受けない."""
        img = _make_color_pixel(0.1, 0.1, 0.9)
        out = _run(session, img, h_center=0.0, h_range=0.1, s_min=0.3, strength=1.0)
        np.testing.assert_allclose(out, img, atol=0.05)

    def test_blue_suppress(self, session):
        """青抑制で青ピクセルが白に."""
        img = _make_color_pixel(0.1, 0.1, 0.9)
        # 青の色相 ≈ 0.67
        out = _run(session, img, h_center=0.67, h_range=0.1, s_min=0.3, strength=1.0)
        np.testing.assert_allclose(out, 1.0, atol=0.05)

    def test_green_suppress(self, session):
        """緑抑制で緑ピクセルが白に."""
        img = _make_color_pixel(0.1, 0.9, 0.1)
        # 緑の色相 ≈ 0.33
        out = _run(session, img, h_center=0.33, h_range=0.1, s_min=0.3, strength=1.0)
        np.testing.assert_allclose(out, 1.0, atol=0.05)

    def test_partial_strength(self, session):
        """strength=0.5 で半分置換."""
        img = _make_color_pixel(0.9, 0.1, 0.1)
        out = _run(session, img, h_center=0.0, h_range=0.1, s_min=0.3, strength=0.5)
        # マッチした場合: out = img * 0.5 + white * 0.5
        expected_r = 0.9 * 0.5 + 1.0 * 0.5  # 0.95
        expected_g = 0.1 * 0.5 + 1.0 * 0.5  # 0.55
        assert out[0, 0, 0, 0] > img[0, 0, 0, 0]  # R は白に近づく
        assert out[0, 1, 0, 0] > img[0, 1, 0, 0]  # G も白に近づく

    def test_mixed_image_selective(self, session):
        """赤/黒混在画像で赤のみ除去、黒は保持."""
        img = np.ones((1, 3, 4, 4), dtype=np.float32) * 0.5
        # 左上に赤ピクセル
        img[0, 0, 0, 0] = 0.9
        img[0, 1, 0, 0] = 0.1
        img[0, 2, 0, 0] = 0.1
        # 右下に黒ピクセル
        img[0, :, 3, 3] = 0.05

        out = _run(session, img, h_center=0.0, h_range=0.1, s_min=0.3, strength=1.0)
        # 赤ピクセルは白に近い
        assert out[0, 0, 0, 0] > 0.9
        # 黒ピクセルは保持
        np.testing.assert_allclose(out[0, :, 3, 3], 0.05, atol=1e-4)

    def test_output_range(self, session):
        """出力値が [0, 1] 範囲内."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img, h_center=0.0, h_range=0.2, s_min=0.2, strength=1.0)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6
