"""直線抽出モデルのテスト.

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

VARIANTS = [("h", 15), ("v", 15)]


@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    for d, L in VARIANTS:
        p = PROJECT_ROOT / "models" / CATEGORY / f"line_extract_{d}_{L}.onnx"
        if not p.exists():
            subprocess.check_call(
                [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
                cwd=str(PROJECT_ROOT),
            )
            break


@pytest.fixture(scope="module", params=VARIANTS, ids=[f"{d}_{L}" for d, L in VARIANTS])
def session_and_variant(request):
    d, L = request.param
    p = PROJECT_ROOT / "models" / CATEGORY / f"line_extract_{d}_{L}.onnx"
    return ort.InferenceSession(str(p)), d, L


def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


class TestLineExtractOutputShape:
    def test_single_image(self, session_and_variant):
        sess, d, L = session_and_variant
        img = np.random.rand(1, 3, 32, 32).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (1, 3, 32, 32)

    def test_batch(self, session_and_variant):
        sess, d, L = session_and_variant
        img = np.random.rand(2, 3, 32, 32).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (2, 3, 32, 32)


class TestLineExtractValues:
    def test_output_range(self, session_and_variant):
        """出力は [0, 1] の範囲内."""
        sess, d, L = session_and_variant
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 64, 64), dtype=np.float32)
        out = _run(sess, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_all_channels_equal(self, session_and_variant):
        """3チャネルは全て同一値."""
        sess, d, L = session_and_variant
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 32, 32), dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        np.testing.assert_array_equal(out[:, 0], out[:, 2])

    def test_uniform_image(self, session_and_variant):
        """均一画像はそのまま出力される (opening は冪等)."""
        sess, d, L = session_and_variant
        img = np.full((1, 3, 32, 32), 0.5, dtype=np.float32)
        out = _run(sess, img)
        # グレースケール値 ≈ 0.5 * sum(luma_weights) が opening 後も維持
        gray_val = float((0.5 * np.array([0.2989, 0.5870, 0.1140])).sum())
        np.testing.assert_allclose(out, gray_val, atol=1e-4)

    def test_horizontal_line_preserved_by_h(self):
        """水平モデルは水平線を保持する."""
        p = PROJECT_ROOT / "models" / CATEGORY / "line_extract_h_15.onnx"
        sess = ort.InferenceSession(str(p))
        img = np.zeros((1, 3, 32, 64), dtype=np.float32)
        # 水平線 (行16, 幅全体)
        img[:, :, 16, :] = 1.0
        out = _run(sess, img)
        # 中央付近の水平線は保持されているはず
        center_val = out[0, 0, 16, 32]
        bg_val = out[0, 0, 0, 32]
        assert center_val > bg_val

    def test_vertical_line_preserved_by_v(self):
        """垂直モデルは垂直線を保持する."""
        p = PROJECT_ROOT / "models" / CATEGORY / "line_extract_v_15.onnx"
        sess = ort.InferenceSession(str(p))
        img = np.zeros((1, 3, 64, 32), dtype=np.float32)
        # 垂直線 (列16, 高さ全体)
        img[:, :, :, 16] = 1.0
        out = _run(sess, img)
        center_val = out[0, 0, 32, 16]
        bg_val = out[0, 0, 32, 0]
        assert center_val > bg_val

    def test_horizontal_line_suppressed_by_v(self):
        """垂直モデルは水平線を抑制する."""
        p = PROJECT_ROOT / "models" / CATEGORY / "line_extract_v_15.onnx"
        sess = ort.InferenceSession(str(p))
        img = np.zeros((1, 3, 64, 64), dtype=np.float32)
        # 水平線のみ
        img[:, :, 32, :] = 1.0
        out = _run(sess, img)
        # 水平線は垂直 opening で消えるはず (背景とほぼ同じ)
        line_val = out[0, 0, 32, 32]
        assert line_val < 0.1
