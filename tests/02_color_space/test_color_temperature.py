"""色温度調整モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "color_temperature.onnx"


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


def _run(session, img: np.ndarray, temperature: float) -> np.ndarray:
    return session.run(None, {
        "input": img,
        "temperature": np.array([temperature], dtype=np.float32),
    })[0]


class TestColorTemperatureOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.0)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.1)
        assert out.shape == (2, 3, 8, 8)


class TestColorTemperatureValues:
    def test_zero_is_identity(self, session):
        """temperature=0 で元画像と同一."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 0.0)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_output_range(self, session):
        """出力は [0, 1] の範囲内."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        for t in [-0.5, -0.2, 0.2, 0.5]:
            out = _run(session, img, t)
            assert out.min() >= -1e-6
            assert out.max() <= 1.0 + 1e-6

    def test_warm_boosts_red(self, session):
        """temperature > 0 で R チャネルが増加."""
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, img, 0.3)
        assert out[0, 0, 0, 0] > 0.5  # R 増加
        assert out[0, 2, 0, 0] < 0.5  # B 減少
        np.testing.assert_allclose(out[:, 1], 0.5, atol=1e-6)  # G 不変

    def test_cool_boosts_blue(self, session):
        """temperature < 0 で B チャネルが増加."""
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, img, -0.3)
        assert out[0, 0, 0, 0] < 0.5  # R 減少
        assert out[0, 2, 0, 0] > 0.5  # B 増加
        np.testing.assert_allclose(out[:, 1], 0.5, atol=1e-6)  # G 不変

    def test_matches_numpy(self, session):
        """NumPy 参照実装と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 8, 8), dtype=np.float32)
        t = 0.2
        gain = np.array([1 + t, 1, 1 - t], dtype=np.float32).reshape(1, 3, 1, 1)
        expected = np.clip(img * gain, 0, 1)
        out = _run(session, img, t)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_clipping_at_extremes(self, session):
        """高い temperature で白画素の R が 1.0 にクリップされること."""
        img = np.ones((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img, 0.5)
        np.testing.assert_allclose(out[:, 0], 1.0, atol=1e-6)  # 1*1.5 → clip 1.0
        np.testing.assert_allclose(out[:, 2], 0.5, atol=1e-6)  # 1*0.5 = 0.5
