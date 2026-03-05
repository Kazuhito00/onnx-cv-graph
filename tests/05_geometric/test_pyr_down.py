"""ピラミッドダウンモデルのテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_PATH = PROJECT_ROOT / "models" / CATEGORY / "pyr_down.onnx"


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


def _run(session, img):
    return session.run(None, {"input": img})[0]


class TestPyrDownOutputShape:
    def test_half_size(self, session):
        """出力が入力の半分のサイズ."""
        img = np.random.rand(1, 3, 32, 32).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (1, 3, 16, 16)

    def test_odd_size(self, session):
        """奇数サイズ入力."""
        img = np.random.rand(1, 3, 33, 33).astype(np.float32)
        out = _run(session, img)
        # Resize linear で scale=0.5: floor(33*0.5) or ceil → 実装依存
        assert out.shape[2] in [16, 17]
        assert out.shape[3] in [16, 17]

    def test_batch(self, session):
        img = np.random.rand(2, 3, 16, 16).astype(np.float32)
        out = _run(session, img)
        assert out.shape == (2, 3, 8, 8)


class TestPyrDownValues:
    def test_uniform_unchanged(self, session):
        """均一画像は縮小後も同じ値."""
        img = np.full((1, 3, 32, 32), 0.5, dtype=np.float32)
        out = _run(session, img)
        np.testing.assert_allclose(out, 0.5, atol=1e-3)

    def test_output_range(self, session):
        """出力値が [0, 1] 範囲内."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 32, 32), dtype=np.float32)
        out = _run(session, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_smoothing_effect(self, session):
        """ぼかし効果: 高周波成分が減衰."""
        img = np.zeros((1, 3, 32, 32), dtype=np.float32)
        # チェッカーボードパターン (高周波)
        img[:, :, ::2, ::2] = 1.0
        img[:, :, 1::2, 1::2] = 1.0
        out = _run(session, img)
        # 出力の分散は入力より小さい (ぼかしにより)
        assert out.var() < img.var()


class TestPyrDownVsOpenCV:
    def test_matches_opencv(self, session):
        """OpenCV pyrDown との近似比較.

        blur→resize パイプラインと OpenCV ネイティブ pyrDown は
        処理順が異なるため完全一致しない。float32 で atol=0.15 を許容.
        """
        rng = np.random.default_rng(123)
        img_nchw = rng.random((1, 3, 32, 32), dtype=np.float32)

        onnx_out = _run(session, img_nchw)

        # OpenCV pyrDown (float32 で実行)
        hwc = img_nchw[0].transpose(1, 2, 0)
        cv_out = cv2.pyrDown(hwc)
        cv_nchw = cv_out.transpose(2, 0, 1)[np.newaxis]

        # サイズが一致する場合のみ比較
        if onnx_out.shape == cv_nchw.shape:
            np.testing.assert_allclose(onnx_out, cv_nchw, atol=0.15)
