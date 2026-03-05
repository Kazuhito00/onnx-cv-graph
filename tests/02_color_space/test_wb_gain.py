"""チャネルゲイン式ホワイトバランスモデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "wb_gain.onnx"


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


def _run(session, img, r_gain=1.0, g_gain=1.0, b_gain=1.0):
    return session.run(None, {
        "input": img,
        "r_gain": np.array([r_gain], dtype=np.float32),
        "g_gain": np.array([g_gain], dtype=np.float32),
        "b_gain": np.array([b_gain], dtype=np.float32),
    })[0]


class TestWbGainOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        assert _run(session, img).shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        assert _run(session, img).shape == (2, 3, 8, 8)


class TestWbGainValues:
    def test_all_one_unchanged(self, session):
        """全ゲイン 1.0 で入力と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 1.0, 1.0, 1.0)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_r_gain_only(self, session):
        """R チャネルのみ 2倍."""
        img = np.full((1, 3, 4, 4), 0.3, dtype=np.float32)
        out = _run(session, img, r_gain=2.0, g_gain=1.0, b_gain=1.0)
        np.testing.assert_allclose(out[0, 0], 0.6, atol=1e-5)  # R
        np.testing.assert_allclose(out[0, 1], 0.3, atol=1e-5)  # G
        np.testing.assert_allclose(out[0, 2], 0.3, atol=1e-5)  # B

    def test_clip_upper(self, session):
        """ゲインで 1.0 を超える場合クリップ."""
        img = np.full((1, 3, 4, 4), 0.6, dtype=np.float32)
        out = _run(session, img, r_gain=3.0)
        np.testing.assert_allclose(out[0, 0], 1.0, atol=1e-6)

    def test_zero_gain(self, session):
        """ゲイン 0 でそのチャネルが 0."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, r_gain=0.0, g_gain=1.0, b_gain=1.0)
        np.testing.assert_allclose(out[0, 0], 0.0, atol=1e-6)

    def test_matches_numpy_reference(self, session):
        """NumPy 参照実装と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        gains = np.array([1.5, 0.8, 1.2], dtype=np.float32).reshape(1, 3, 1, 1)
        expected = np.clip(img * gains, 0.0, 1.0)
        out = _run(session, img, 1.5, 0.8, 1.2)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_output_range(self, session):
        """出力値が [0, 1] 範囲内."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img, 3.0, 3.0, 3.0)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6
