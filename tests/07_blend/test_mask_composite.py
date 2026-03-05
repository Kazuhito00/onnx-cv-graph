"""マスク合成モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "mask_composite.onnx"


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
    """ONNX Runtime の推論セッションを返す."""
    return ort.InferenceSession(str(MODEL_PATH))


def _run(session, img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return session.run(None, {
        "input": img1,
        "input2": img2,
        "mask": mask,
    })[0]


def _numpy_mask_composite(img1, img2, mask):
    """NumPy 参照実装."""
    return mask * img1 + (1 - mask) * img2


class TestMaskCompositeOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        mask = np.random.rand(1, 1, 8, 8).astype(np.float32)
        out = _run(session, img, img, mask)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        mask = np.random.rand(2, 1, 8, 8).astype(np.float32)
        out = _run(session, img, img, mask)
        assert out.shape == (2, 3, 8, 8)


class TestMaskCompositeValues:
    """出力値の正確性を検証するテスト群."""

    def test_all_ones_mask(self, session):
        """マスク全1で input がそのまま出力されること."""
        img1 = np.full((1, 3, 4, 4), 0.3, dtype=np.float32)
        img2 = np.full((1, 3, 4, 4), 0.7, dtype=np.float32)
        mask = np.ones((1, 1, 4, 4), dtype=np.float32)
        out = _run(session, img1, img2, mask)
        np.testing.assert_allclose(out, 0.3, atol=1e-5)

    def test_all_zeros_mask(self, session):
        """マスク全0で input2 がそのまま出力されること."""
        img1 = np.full((1, 3, 4, 4), 0.3, dtype=np.float32)
        img2 = np.full((1, 3, 4, 4), 0.7, dtype=np.float32)
        mask = np.zeros((1, 1, 4, 4), dtype=np.float32)
        out = _run(session, img1, img2, mask)
        np.testing.assert_allclose(out, 0.7, atol=1e-5)

    def test_half_mask(self, session):
        """マスク 0.5 で平均値."""
        img1 = np.full((1, 3, 4, 4), 0.2, dtype=np.float32)
        img2 = np.full((1, 3, 4, 4), 0.8, dtype=np.float32)
        mask = np.full((1, 1, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, img1, img2, mask)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_spatial_mask(self, session):
        """空間的に異なるマスクで左右が異なる入力を選択すること."""
        img1 = np.full((1, 3, 4, 4), 1.0, dtype=np.float32)
        img2 = np.zeros((1, 3, 4, 4), dtype=np.float32)
        mask = np.zeros((1, 1, 4, 4), dtype=np.float32)
        mask[:, :, :, :2] = 1.0  # 左半分は img1
        out = _run(session, img1, img2, mask)
        np.testing.assert_allclose(out[:, :, :, :2], 1.0, atol=1e-5)
        np.testing.assert_allclose(out[:, :, :, 2:], 0.0, atol=1e-5)

    def test_matches_numpy_reference(self, session):
        """ランダム入力で NumPy 参照実装と一致すること."""
        rng = np.random.default_rng(42)
        img1 = rng.random((2, 3, 16, 16), dtype=np.float32)
        img2 = rng.random((2, 3, 16, 16), dtype=np.float32)
        mask = rng.random((2, 1, 16, 16), dtype=np.float32)
        expected = _numpy_mask_composite(img1, img2, mask)
        out = _run(session, img1, img2, mask)
        np.testing.assert_allclose(out, expected, atol=1e-5)
