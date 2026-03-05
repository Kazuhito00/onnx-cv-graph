"""任意角度回転 (rotate_arbitrary) モデルのテスト.

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
MODEL_DIR = PROJECT_ROOT / "models" / CATEGORY


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    needed = [
        MODEL_DIR / "rotate_arbitrary.onnx",
        MODEL_DIR / "rotate_90.onnx",
        MODEL_DIR / "rotate_180.onnx",
    ]
    if any(not p.exists() for p in needed):
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL_DIR / "rotate_arbitrary.onnx"))


def _run(session, img: np.ndarray, angle: float) -> np.ndarray:
    return session.run(None, {
        "input": img,
        "angle": np.array([angle], dtype=np.float32),
    })[0]


class TestRotateArbitraryOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        img = np.random.rand(1, 3, 16, 20).astype(np.float32)
        out = _run(session, img, 45.0)
        assert out.shape == (1, 3, 16, 20)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 12).astype(np.float32)
        out = _run(session, img, 30.0)
        assert out.shape == (2, 3, 8, 12)


class TestRotateArbitraryValues:
    """出力値の正確性を検証するテスト群."""

    def test_angle_zero_identity(self, session):
        """angle=0 のとき入力と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img, 0.0)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_uniform_image_unchanged(self, session):
        """均一画像は回転後も変化しないこと (はみ出しのない正方形の場合)."""
        img = np.full((1, 3, 16, 16), 0.5, dtype=np.float32)
        out = _run(session, img, 30.0)
        # 角の部分は zeros パディングで 0 になるためチェックしない
        # 中央付近のみ検証
        center = out[:, :, 4:12, 4:12]
        np.testing.assert_allclose(center, 0.5, atol=1e-5)

    def test_angle_90_matches_rotate90(self, session):
        """angle=90 が Rotate90Op と一致すること (正方形画像)."""
        sess90 = ort.InferenceSession(str(MODEL_DIR / "rotate_90.onnx"))
        rng = np.random.default_rng(100)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)

        out_arb = _run(session, img, 90.0)
        out_90 = sess90.run(None, {"input": img})[0]
        np.testing.assert_allclose(out_arb, out_90, atol=1e-5)

    def test_angle_180_matches_rotate180(self, session):
        """angle=180 が Rotate180Op と一致すること."""
        sess180 = ort.InferenceSession(str(MODEL_DIR / "rotate_180.onnx"))
        rng = np.random.default_rng(200)
        img = rng.random((1, 3, 16, 20), dtype=np.float32)

        out_arb = _run(session, img, 180.0)
        out_180 = sess180.run(None, {"input": img})[0]
        np.testing.assert_allclose(out_arb, out_180, atol=1e-5)

    def test_angle_neg180_matches_rotate180(self, session):
        """angle=-180 が Rotate180Op と一致すること."""
        sess180 = ort.InferenceSession(str(MODEL_DIR / "rotate_180.onnx"))
        rng = np.random.default_rng(300)
        img = rng.random((1, 3, 12, 12), dtype=np.float32)

        out_arb = _run(session, img, -180.0)
        out_180 = sess180.run(None, {"input": img})[0]
        np.testing.assert_allclose(out_arb, out_180, atol=1e-5)
