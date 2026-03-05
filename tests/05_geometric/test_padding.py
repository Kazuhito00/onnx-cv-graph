"""パディングモデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "padding_reflect.onnx"


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


def _run(session, img, pad_ratio=0.1):
    r = np.array([pad_ratio], dtype=np.float32)
    return session.run(None, {"input": img, "pad_ratio": r})[0]


class TestPaddingOutputShape:
    def test_ratio_zero_same_size(self, session):
        """pad_ratio=0 で入力と同じサイズ."""
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        out = _run(session, img, 0.0)
        assert out.shape == (1, 3, 16, 16)

    def test_ratio_adds_pixels(self, session):
        """pad_ratio=0.25 で H,W が 25% ずつ上下左右に追加."""
        img = np.random.rand(1, 3, 16, 20).astype(np.float32)
        out = _run(session, img, 0.25)
        # pad_h = floor(16*0.25)=4, pad_w = floor(20*0.25)=5
        assert out.shape == (1, 3, 16 + 4 * 2, 20 + 5 * 2)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.25)
        assert out.shape[0] == 2
        assert out.shape[1] == 3


class TestPaddingValues:
    def test_center_preserved(self, session):
        """中央部分は元画像と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 20), dtype=np.float32)
        out = _run(session, img, 0.25)
        pad_h, pad_w = 4, 5
        center = out[:, :, pad_h:-pad_h, pad_w:-pad_w]
        np.testing.assert_allclose(center, img, atol=1e-6)

    def test_reflect_padding(self, session):
        """reflect パディングで端の値が鏡像反転."""
        img = np.zeros((1, 3, 8, 8), dtype=np.float32)
        img[0, 0, 0, :] = 1.0  # 上端行 (index=0) を白に
        out = _run(session, img, 0.25)
        pad_h = 2  # floor(8*0.25)
        # reflect: パディング領域の行 pad_h-1 は入力の行 1 を反射
        # 入力行 0 = 1.0 → reflect で out[pad_h, :] = 1.0 (入力行0)
        assert out[0, 0, pad_h, pad_h] == 1.0  # 入力行0がそのまま
        # パディング領域の行 pad_h-1 は入力行 1 の反射 (= 0.0)
        assert out[0, 0, pad_h - 1, pad_h] == 0.0

    def test_uniform_unchanged_content(self, session):
        """均一画像はパディング部分も同じ値."""
        img = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)
        out = _run(session, img, 0.25)
        np.testing.assert_allclose(out, 0.5, atol=1e-6)
