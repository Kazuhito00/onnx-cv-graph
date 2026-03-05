"""ポスタリゼーションモデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "posterize.onnx"


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


def _run(session, img: np.ndarray, levels: float) -> np.ndarray:
    lv = np.array([levels], dtype=np.float32)
    return session.run(None, {"input": img, "levels": lv})[0]


def _numpy_posterize(img: np.ndarray, levels: float) -> np.ndarray:
    """NumPy 参照実装."""
    return np.clip(np.floor(img * levels) / levels, 0.0, 1.0).astype(np.float32)


class TestPosterizeOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 4.0)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 4.0)
        assert out.shape == (2, 3, 8, 8)


class TestPosterizeValues:
    def test_levels_2_binary(self, session):
        """levels=2 で 0.0 or 0.5 の2値に量子化."""
        img = np.array([[[[0.1, 0.6], [0.3, 0.9]]]], dtype=np.float32)
        img = np.broadcast_to(img, (1, 3, 2, 2)).copy()
        out = _run(session, img, 2.0)
        # floor(0.1*2)/2=0, floor(0.6*2)/2=0.5, floor(0.3*2)/2=0, floor(0.9*2)/2=0.5
        expected = np.array([[[[0.0, 0.5], [0.0, 0.5]]]], dtype=np.float32)
        expected = np.broadcast_to(expected, (1, 3, 2, 2)).copy()
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_levels_4(self, session):
        """levels=4 で 0.0, 0.25, 0.5, 0.75 の4段階."""
        img = np.full((1, 3, 4, 4), 0.6, dtype=np.float32)
        out = _run(session, img, 4.0)
        # floor(0.6*4)/4 = floor(2.4)/4 = 2/4 = 0.5
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_black_unchanged(self, session):
        """黒 (0.0) は変化なし."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img, 4.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        for lv in [2.0, 4.0, 8.0, 16.0]:
            expected = _numpy_posterize(img, lv)
            out = _run(session, img, lv)
            np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_output_range(self, session):
        """出力値が [0, 1] 範囲内."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        for lv in [2.0, 4.0, 32.0]:
            out = _run(session, img, lv)
            assert out.min() >= -1e-6
            assert out.max() <= 1.0 + 1e-6
