"""露出調整モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "exposure.onnx"


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


def _run(session, img: np.ndarray, exposure: float) -> np.ndarray:
    e = np.array([exposure], dtype=np.float32)
    return session.run(None, {"input": img, "exposure": e})[0]


def _numpy_exposure(img: np.ndarray, exposure: float) -> np.ndarray:
    """NumPy 参照実装."""
    return np.clip(img * exposure, 0.0, 1.0).astype(np.float32)


class TestExposureOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 1.0)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 1.0)
        assert out.shape == (2, 3, 8, 8)


class TestExposureValues:
    def test_exposure_one_unchanged(self, session):
        """exposure=1.0 で入力と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 1.0)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_exposure_zero_black(self, session):
        """exposure=0 で全黒."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img, 0.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_exposure_double(self, session):
        """exposure=2.0 で値が2倍 (クリップ)."""
        img = np.full((1, 3, 4, 4), 0.3, dtype=np.float32)
        out = _run(session, img, 2.0)
        np.testing.assert_allclose(out, 0.6, atol=1e-5)

    def test_clip_upper(self, session):
        """大きな exposure で上限クリップ."""
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, img, 3.0)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        for e in [0.0, 0.5, 1.0, 2.0, 5.0]:
            expected = _numpy_exposure(img, e)
            out = _run(session, img, e)
            np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_output_range(self, session):
        """出力値が [0, 1] 範囲内."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        for e in [0.0, 1.0, 5.0]:
            out = _run(session, img, e)
            assert out.min() >= -1e-6
            assert out.max() <= 1.0 + 1e-6
