"""チャネル抽出モデルのテスト.

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

CHANNELS = [("r", 0), ("g", 1), ("b", 2)]


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    for ch, _ in CHANNELS:
        p = PROJECT_ROOT / "models" / CATEGORY / f"channel_{ch}.onnx"
        if not p.exists():
            subprocess.check_call(
                [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
                cwd=str(PROJECT_ROOT),
            )
            break


@pytest.fixture(scope="module", params=CHANNELS, ids=[ch for ch, _ in CHANNELS])
def session_and_idx(request):
    ch, idx = request.param
    p = PROJECT_ROOT / "models" / CATEGORY / f"channel_{ch}.onnx"
    return ort.InferenceSession(str(p)), idx


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


class TestChannelExtractOutputShape:
    def test_single_image(self, session_and_idx):
        sess, idx = session_and_idx
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session_and_idx):
        sess, idx = session_and_idx
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (2, 3, 8, 8)


class TestChannelExtractValues:
    def test_extracts_correct_channel(self, session_and_idx):
        """指定チャネルの値が正しく抽出されること."""
        sess, idx = session_and_idx
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 8, 8), dtype=np.float32)
        out = _run(sess, img)
        expected = img[:, idx:idx+1]
        np.testing.assert_allclose(out[:, 0:1], expected, atol=1e-6)

    def test_all_channels_equal(self, session_and_idx):
        """3チャネル全て同一値 (1ch → 3ch 複製)."""
        sess, idx = session_and_idx
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        np.testing.assert_array_equal(out[:, 0], out[:, 2])

    def test_output_range(self, session_and_idx):
        """出力は [0, 1] の範囲内."""
        sess, idx = session_and_idx
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(sess, img)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_pure_red(self):
        """純赤画像で R=1, G=0, B=0 の抽出確認."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        img[:, 0] = 1.0  # R=1
        for ch, idx in CHANNELS:
            p = PROJECT_ROOT / "models" / CATEGORY / f"channel_{ch}.onnx"
            sess = ort.InferenceSession(str(p))
            out = _run(sess, img)
            expected_val = 1.0 if idx == 0 else 0.0
            np.testing.assert_allclose(out, expected_val, atol=1e-6)
