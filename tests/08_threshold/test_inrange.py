"""範囲内抽出モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "inrange.onnx"

LUMA_WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    if not MODEL_PATH.exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )
    assert MODEL_PATH.exists(), f"モデルが見つかりません: {MODEL_PATH}"


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_PATH))


def _run(session, img: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return session.run(None, {
        "input": img,
        "lower": np.array([lower], dtype=np.float32),
        "upper": np.array([upper], dtype=np.float32),
    })[0]


def _numpy_inrange(img: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """NumPy 参照実装."""
    gray = (img * LUMA_WEIGHTS).sum(axis=1, keepdims=True)
    mask = ((gray >= lower) & (gray <= upper)).astype(np.float32)
    return np.repeat(mask, 3, axis=1)


class TestInrangeOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.2, 0.8)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.2, 0.8)
        assert out.shape == (2, 3, 8, 8)


class TestInrangeValues:
    def test_output_only_zero_or_one(self, session):
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        out = _run(session, img, 0.3, 0.7)
        unique = np.unique(out)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_all_in_range(self, session):
        """全画素が範囲内なら全 1."""
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, img, 0.0, 1.0)
        np.testing.assert_array_equal(out, 1.0)

    def test_all_out_of_range(self, session):
        """全画素が範囲外なら全 0."""
        img = np.full((1, 3, 4, 4), 0.1, dtype=np.float32)
        out = _run(session, img, 0.5, 1.0)
        np.testing.assert_array_equal(out, 0.0)

    def test_boundary_inclusive(self, session):
        """lower/upper ちょうどの値は範囲内 (inclusive)."""
        # gray = sum(0.5 * weights) ≈ 0.4999... なので lower/upper もそれに合わせる
        gray_val = float((np.full((1, 3, 1, 1), 0.5, dtype=np.float32) * LUMA_WEIGHTS).sum())
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, img, gray_val - 0.01, gray_val + 0.01)
        np.testing.assert_array_equal(out, 1.0)

    def test_all_channels_equal(self, session):
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 0.3, 0.7)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        np.testing.assert_array_equal(out[:, 0], out[:, 2])

    def test_matches_numpy_reference(self, session):
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        expected = _numpy_inrange(img, 0.3, 0.7)
        out = _run(session, img, 0.3, 0.7)
        np.testing.assert_array_equal(out, expected)
