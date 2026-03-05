"""彩度調整モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "saturation.onnx"


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    if not MODEL_PATH.exists():
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )
    assert MODEL_PATH.exists(), f"モデルが見つかりません: {MODEL_PATH}"


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_PATH))


def _run(session, img: np.ndarray, saturation: float) -> np.ndarray:
    s = np.array([saturation], dtype=np.float32)
    return session.run(None, {"input": img, "saturation": s})[0]


def _numpy_saturation(img: np.ndarray, saturation: float) -> np.ndarray:
    """NumPy 参照実装."""
    luma = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
    gray = (img * luma).sum(axis=1, keepdims=True)
    gray = np.broadcast_to(gray, img.shape).copy()
    return np.clip(gray * (1 - saturation) + img * saturation, 0.0, 1.0).astype(np.float32)


class TestSaturationOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 1.0)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 1.0)
        assert out.shape == (2, 3, 8, 8)


class TestSaturationValues:
    def test_saturation_one_unchanged(self, session):
        """saturation=1.0 で入力と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 1.0)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_saturation_zero_grayscale(self, session):
        """saturation=0 でグレースケール."""
        img = np.array([[[[1.0]], [[0.0]], [[0.0]]]], dtype=np.float32)  # 純赤
        out = _run(session, img, 0.0)
        # グレー = 0.2989
        expected = np.full((1, 3, 1, 1), 0.2989, dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-4)

    def test_already_gray_unchanged(self, session):
        """既にグレーの画像はどの saturation でもほぼ変化なし.

        luma 重み合計が厳密に 1.0 でないため微小誤差あり.
        """
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        for s in [0.0, 1.0, 2.0]:
            out = _run(session, img, s)
            np.testing.assert_allclose(out, 0.5, atol=1e-3)

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        for s in [0.0, 0.5, 1.0, 1.5, 2.0]:
            expected = _numpy_saturation(img, s)
            out = _run(session, img, s)
            np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_output_range(self, session):
        """出力値が [0, 1] 範囲内."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        for s in [0.0, 1.0, 3.0]:
            out = _run(session, img, s)
            assert out.min() >= -1e-6
            assert out.max() <= 1.0 + 1e-6
