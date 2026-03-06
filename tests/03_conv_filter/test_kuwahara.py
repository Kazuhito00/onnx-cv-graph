"""Kuwahara フィルタモデルのテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
5×5 / 7×7 / 9×9 の全バリアントをテストする.

OpenCV に Kuwahara フィルタの相当 API が無いため VsOpenCV カテゴリは省略し、
代わりに NumPy 参照実装との比較で正確性を担保する.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATEGORY = Path(__file__).resolve().parent.name
MODEL_DIR = PROJECT_ROOT / "models" / CATEGORY

KERNEL_SIZES = [5, 7, 9]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    missing = any(
        not (MODEL_DIR / f"kuwahara_{k}x{k}.onnx").exists() for k in KERNEL_SIZES
    )
    if missing:
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )
    for k in KERNEL_SIZES:
        p = MODEL_DIR / f"kuwahara_{k}x{k}.onnx"
        assert p.exists(), f"モデルが見つかりません: {p}"


@pytest.fixture(
    scope="module",
    params=KERNEL_SIZES,
    ids=[f"sess_{k}x{k}" for k in KERNEL_SIZES],
)
def session_and_k(request):
    """(InferenceSession, kernel_size) のタプルを返す."""
    k = request.param
    sess = ort.InferenceSession(str(MODEL_DIR / f"kuwahara_{k}x{k}.onnx"))
    return sess, k


# ---------------------------------------------------------------------------
# Helper: 推論実行
# ---------------------------------------------------------------------------

def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


# ---------------------------------------------------------------------------
# Helper: NumPy 参照実装
# ---------------------------------------------------------------------------

def _kuwahara_numpy(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """Kuwahara フィルタの NumPy 参照実装.

    img: (N, 3, H, W) float32
    各ピクセルについて4象限の輝度分散を計算し、最小分散象限の RGB 平均を返す.
    ONNX の AveragePool(count_include_pad=0) と同じ境界処理を採用する.
    """
    img = img.astype(np.float32)
    N, C, H, W = img.shape
    r = (kernel_size - 1) // 2
    sub = r + 1
    output = np.zeros_like(img)

    # 各象限のサブ領域オフセット (上端・左端の相対位置)
    quadrant_offsets = [
        (-r, -r),  # Q1: 左上
        (-r,  0),  # Q2: 右上
        ( 0, -r),  # Q3: 左下
        ( 0,  0),  # Q4: 右下
    ]

    for n in range(N):
        # ITU-R BT.601 輝度 (H, W)
        luma = (0.2989 * img[n, 0] + 0.5870 * img[n, 1] + 0.1140 * img[n, 2])

        for y in range(H):
            for x in range(W):
                best_var = np.inf
                best_mean = img[n, :, y, x].copy()

                for dy, dx in quadrant_offsets:
                    y0 = max(0, y + dy)
                    y1 = min(H, y + dy + sub)
                    x0 = max(0, x + dx)
                    x1 = min(W, x + dx + sub)

                    region = luma[y0:y1, x0:x1].astype(np.float32)
                    m = region.mean()
                    var = (region * region).mean() - m * m  # E[X²] - E[X]²

                    if var < best_var:
                        best_var = var
                        best_mean = img[n, :, y0:y1, x0:x1].mean(axis=(1, 2))

                output[n, :, y, x] = best_mean

    return np.clip(output, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 1. 形状テスト
# ---------------------------------------------------------------------------

class TestKuwaharaOutputShape:

    def test_single_image(self, session_and_k):
        """単一画像 (N=1) で出力が (1,3,H,W) になること."""
        sess, _ = session_and_k
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (1, 3, 16, 16)

    def test_batch(self, session_and_k):
        """バッチ (N>1) で出力が (N,3,H,W) になること."""
        sess, _ = session_and_k
        img = np.random.rand(2, 3, 16, 16).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (2, 3, 16, 16)

    def test_rectangular(self, session_and_k):
        """非正方形画像で出力形状が入力と同じであること."""
        sess, _ = session_and_k
        img = np.random.rand(1, 3, 12, 20).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (1, 3, 12, 20)


# ---------------------------------------------------------------------------
# 2. 値テスト
# ---------------------------------------------------------------------------

class TestKuwaharaValues:

    def test_uniform_image_unchanged(self, session_and_k):
        """均一画像はフィルタ後も値が変化しないこと."""
        sess, _ = session_and_k
        img = np.full((1, 3, 16, 16), 0.5, dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_all_black(self, session_and_k):
        """全黒画像は全黒のまま."""
        sess, _ = session_and_k
        img = np.zeros((1, 3, 16, 16), dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_array_equal(out, 0.0)

    def test_all_white(self, session_and_k):
        """全白画像は全白のまま."""
        sess, _ = session_and_k
        img = np.ones((1, 3, 16, 16), dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_allclose(out, 1.0, atol=1e-5)

    def test_output_range(self, session_and_k):
        """出力値が [0, 1] 範囲内であること."""
        sess, _ = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16)).astype(np.float32)
        out = _run(sess, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_no_nan_inf(self, session_and_k):
        """出力に NaN / Inf が含まれないこと."""
        sess, _ = session_and_k
        rng = np.random.default_rng(7)
        img = rng.random((1, 3, 16, 16)).astype(np.float32)
        out = _run(sess, img)
        assert np.all(np.isfinite(out))

    def test_matches_numpy_reference(self, session_and_k):
        """ランダム入力で NumPy 参照実装と一致すること (atol=1e-4).

        ONNX の AveragePool は float32 で演算するため、
        NumPy float64 内部計算との誤差を考慮して atol=1e-4 を許容する.
        """
        sess, k = session_and_k
        rng = np.random.default_rng(42)
        # 参照実装が O(H×W) の逐次処理のため小さい画像で検証
        img = rng.random((1, 3, 12, 12)).astype(np.float32)
        expected = _kuwahara_numpy(img, k)
        out = _run(sess, img)
        np.testing.assert_allclose(out, expected, atol=1e-4)

    def test_edge_preservation(self, session_and_k):
        """左右に明確なエッジがある画像でエッジが保持されること.

        左半分が黒、右半分が白の画像に対して、
        Kuwahara フィルタはエッジを保持するため中間値が少ないはず.
        """
        sess, _ = session_and_k
        img = np.zeros((1, 3, 24, 24), dtype=np.float32)
        img[:, :, :, 12:] = 1.0  # 右半分を白
        out = _run(sess, img)
        # エッジから十分離れた左端・右端は元の値に近いこと
        np.testing.assert_allclose(out[0, :, :, :4], 0.0, atol=0.1)
        np.testing.assert_allclose(out[0, :, :, 20:], 1.0, atol=0.1)


# ---------------------------------------------------------------------------
# 3. モデル整合性テスト
# ---------------------------------------------------------------------------

class TestKuwaharaModelIntegrity:

    @pytest.mark.parametrize("k", KERNEL_SIZES)
    def test_onnx_checker(self, k):
        """onnx.checker がエラーなく通ること."""
        model = onnx.load(str(MODEL_DIR / f"kuwahara_{k}x{k}.onnx"))
        onnx.checker.check_model(model)

    @pytest.mark.parametrize("k", KERNEL_SIZES)
    def test_opset_version(self, k):
        """opset version が 17 であること."""
        model = onnx.load(str(MODEL_DIR / f"kuwahara_{k}x{k}.onnx"))
        opset = model.opset_import[0].version
        assert opset == 17, f"opset={opset}"

    @pytest.mark.parametrize("k", KERNEL_SIZES)
    def test_io_names(self, k):
        """入出力名が仕様通りであること."""
        model = onnx.load(str(MODEL_DIR / f"kuwahara_{k}x{k}.onnx"))
        inputs = [i.name for i in model.graph.input
                  if i.name not in {init.name for init in model.graph.initializer}]
        outputs = [o.name for o in model.graph.output]
        assert inputs == ["input"]
        assert outputs == ["output"]
