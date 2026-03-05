"""Shi-Tomasi コーナースコアモデルのテスト.

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

BLOCK_SIZES = [3, 5]


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    for bs in BLOCK_SIZES:
        p = PROJECT_ROOT / "models" / CATEGORY / f"shi_tomasi_{bs}x{bs}.onnx"
        if not p.exists():
            subprocess.check_call(
                [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
                cwd=str(PROJECT_ROOT),
            )
            break


@pytest.fixture(scope="module", params=BLOCK_SIZES, ids=[f"bs{bs}" for bs in BLOCK_SIZES])
def session_and_bs(request):
    bs = request.param
    p = PROJECT_ROOT / "models" / CATEGORY / f"shi_tomasi_{bs}x{bs}.onnx"
    return ort.InferenceSession(str(p)), bs


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


class TestShiTomasiOutputShape:
    def test_single_image(self, session_and_bs):
        sess, bs = session_and_bs
        img = np.random.rand(1, 3, 32, 32).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (1, 3, 32, 32)

    def test_batch(self, session_and_bs):
        sess, bs = session_and_bs
        img = np.random.rand(2, 3, 32, 32).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (2, 3, 32, 32)


class TestShiTomasiValues:
    def test_output_range(self, session_and_bs):
        """出力は [0, 1] の範囲内."""
        sess, bs = session_and_bs
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 32, 32), dtype=np.float32)
        out = _run(sess, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_all_channels_equal(self, session_and_bs):
        """3チャネルは全て同一値."""
        sess, bs = session_and_bs
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 32, 32), dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        np.testing.assert_array_equal(out[:, 0], out[:, 2])

    def test_uniform_image_no_features(self, session_and_bs):
        """均一画像ではスコアがほぼ 0."""
        sess, bs = session_and_bs
        img = np.full((1, 3, 32, 32), 0.5, dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_corner_response_at_corner(self, session_and_bs):
        """L字コーナーで応答が高くなること."""
        sess, bs = session_and_bs
        img = np.zeros((1, 3, 32, 32), dtype=np.float32)
        img[:, :, 4:16, 4:16] = 1.0
        out = _run(sess, img)
        corner_vals = [out[0, 0, r, c] for r, c in [(4, 4), (4, 15), (15, 4), (15, 15)]]
        assert max(corner_vals) > 0.0

    def test_edge_response_less_than_corner(self, session_and_bs):
        """エッジのスコアはコーナーより低いこと."""
        sess, bs = session_and_bs
        img = np.zeros((1, 3, 32, 32), dtype=np.float32)
        img[:, :, 4:16, 4:16] = 1.0
        out = _run(sess, img)
        # コーナー (角) の応答
        corner_max = max(out[0, 0, r, c] for r, c in [(4, 4), (4, 15), (15, 4), (15, 15)])
        # エッジ (辺の中央) の応答
        edge_val = out[0, 0, 10, 4]  # 左辺の中央付近
        # Shi-Tomasi ではコーナーの最小固有値 > エッジの最小固有値
        assert corner_max >= edge_val
