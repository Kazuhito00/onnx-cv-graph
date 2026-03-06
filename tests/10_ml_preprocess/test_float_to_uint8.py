"""float32→uint8 量子化モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "float_to_uint8.onnx"


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


def _run(session, img):
    return session.run(None, {"input": img})[0]


class TestFloatToUint8OutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (2, 3, 8, 8)


class TestFloatToUint8Dtype:
    def test_output_is_uint8(self, session):
        img = np.random.rand(1, 3, 4, 4).astype(np.float32)
        out = _run(session, img)
        assert out.dtype == np.uint8


class TestFloatToUint8Values:
    def test_zero_to_zero(self, session):
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img)
        assert (out == 0).all()

    def test_one_to_255(self, session):
        img = np.ones((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img)
        assert (out == 255).all()

    def test_half_to_128(self, session):
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, img)
        # 0.5 * 255 = 127.5 → round → 128
        assert (out == 128).all()

    def test_clipping_above(self, session):
        """1.0 を超える値は 255 にクリップ."""
        img = np.full((1, 3, 4, 4), 1.5, dtype=np.float32)
        out = _run(session, img)
        assert (out == 255).all()

    def test_clipping_below(self, session):
        """0.0 未満の値は 0 にクリップ."""
        img = np.full((1, 3, 4, 4), -0.5, dtype=np.float32)
        out = _run(session, img)
        assert (out == 0).all()

    def test_matches_numpy(self, session):
        """NumPy 相当の変換と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        expected = np.round(img * 255.0).clip(0, 255).astype(np.uint8)
        np.testing.assert_array_equal(out, expected)

    def test_roundtrip(self, session):
        """float→uint8→float で元に近い値."""
        rev_path = PROJECT_ROOT / "models" / CATEGORY / "uint8_to_float.onnx"
        if not rev_path.exists():
            pytest.skip("uint8_to_float モデルが見つかりません")
        rev_sess = ort.InferenceSession(str(rev_path))

        rng = np.random.default_rng(99)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        u8 = _run(session, img)
        restored = rev_sess.run(None, {"input": u8})[0]
        # 量子化誤差: 最大 0.5/255 ≈ 0.002
        np.testing.assert_allclose(restored, img, atol=0.003)

