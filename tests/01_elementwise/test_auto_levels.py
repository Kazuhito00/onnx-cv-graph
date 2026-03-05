"""チャネル別オートレベル補正モデルのテスト.

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
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "auto_levels.onnx"


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


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


def auto_levels_numpy(img: np.ndarray) -> np.ndarray:
    """NumPy 参照実装."""
    ch_min = img.min(axis=(2, 3), keepdims=True)
    ch_max = img.max(axis=(2, 3), keepdims=True)
    ch_range = ch_max - ch_min
    ch_range = np.where(ch_range == 0, 1e-8, ch_range)
    result = (img - ch_min) / ch_range
    return np.clip(result, 0, 1)


class TestAutoLevelsOutputShape:
    def test_single_image(self, session):
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        img = np.random.rand(2, 3, 8, 8).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (2, 3, 8, 8)


class TestAutoLevelsValues:
    def test_output_range_per_channel(self, session):
        """各チャネルの min ≈ 0, max ≈ 1."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16), dtype=np.float32)
        out = _run(session, img)
        for c in range(3):
            ch = out[0, c]
            assert ch.min() == pytest.approx(0.0, abs=1e-5)
            assert ch.max() == pytest.approx(1.0, abs=1e-5)

    def test_channel_independence(self, session):
        """各チャネルが異なるレンジの入力 → チャネル独立に正規化."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        img[0, 0] = np.linspace(0.2, 0.4, 16).reshape(4, 4)  # R: [0.2, 0.4]
        img[0, 1] = np.linspace(0.5, 0.9, 16).reshape(4, 4)  # G: [0.5, 0.9]
        img[0, 2] = np.linspace(0.0, 1.0, 16).reshape(4, 4)  # B: [0.0, 1.0]
        out = _run(session, img)
        for c in range(3):
            ch = out[0, c]
            assert ch.min() == pytest.approx(0.0, abs=1e-5)
            assert ch.max() == pytest.approx(1.0, abs=1e-5)

    def test_uniform_channel(self, session):
        """1チャネルが定数 → そのチャネルは ≈ 0 (ε 除算)、他チャネルは正常."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        img[0, 1, :, :] = 0.5  # G チャネルを定数に
        out = _run(session, img)
        # 定数チャネルは 0 付近
        assert np.allclose(out[0, 1], 0.0, atol=1e-2)
        # 他チャネルは正常に正規化
        assert out[0, 0].min() == pytest.approx(0.0, abs=1e-5)
        assert out[0, 0].max() == pytest.approx(1.0, abs=1e-5)

    def test_already_normalized_input(self, session):
        """既に [0,1] の入力 → 出力も [0,1] 範囲."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 8, 8), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_output_range_random(self, session):
        """ランダム入力で出力 [0, 1] 保証."""
        rng = np.random.default_rng(123)
        img = rng.random((4, 3, 32, 32), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_matches_numpy_reference(self, session):
        """NumPy 参照実装と一致."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        expected = auto_levels_numpy(img)
        out = _run(session, img)
        np.testing.assert_allclose(out, expected, atol=1e-5)
