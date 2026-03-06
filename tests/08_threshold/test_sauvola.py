"""Sauvola 局所適応2値化モデルのテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
15×15 / 31×31 / 63×63 の全バリアントをテストする.

閾値式: T = μ × [1 + k × (σ/R − 1)]  (R=0.5 固定、k は推論時入力)
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

KERNEL_SIZES = [15, 31, 63]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    missing = any(
        not (MODEL_DIR / f"sauvola_{k}x{k}.onnx").exists() for k in KERNEL_SIZES
    )
    if missing:
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )
    for k in KERNEL_SIZES:
        p = MODEL_DIR / f"sauvola_{k}x{k}.onnx"
        assert p.exists(), f"モデルが見つかりません: {p}"


@pytest.fixture(
    scope="module",
    params=KERNEL_SIZES,
    ids=[f"sess_{k}x{k}" for k in KERNEL_SIZES],
)
def session_and_k(request):
    """(InferenceSession, kernel_size) のタプルを返す."""
    k = request.param
    sess = ort.InferenceSession(str(MODEL_DIR / f"sauvola_{k}x{k}.onnx"))
    return sess, k


# ---------------------------------------------------------------------------
# Helper: 推論実行
# ---------------------------------------------------------------------------

def _run(session, img: np.ndarray, k_val: float = 0.5) -> np.ndarray:
    k_input = np.array([k_val], dtype=np.float32)
    return session.run(None, {"input": img, "k": k_input})[0]


# ---------------------------------------------------------------------------
# Helper: NumPy 参照実装
# ---------------------------------------------------------------------------

def _sauvola_numpy(img: np.ndarray, kernel_size: int, k: float = 0.5, R: float = 0.5) -> np.ndarray:
    """Sauvola 二値化の NumPy 参照実装.

    img: (N, 3, H, W) float32
    ONNX グラフと同じ輝度係数・AveragePool(reflect pad) を使用する.
    """
    img = img.astype(np.float32)
    N, C, H, W = img.shape
    pad = kernel_size // 2

    # グレースケール変換 (N,1,H,W)
    luma = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
    gray = (img * luma).sum(axis=1, keepdims=True)  # (N,1,H,W)

    # reflect padding
    gray_pad = np.pad(gray, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")
    gray2_pad = np.pad(gray ** 2, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")

    # sliding window average (AveragePool 相当)
    output = np.zeros((N, 3, H, W), dtype=np.float32)
    for n in range(N):
        for i in range(H):
            for j in range(W):
                region = gray_pad[n, 0, i:i + kernel_size, j:j + kernel_size]
                region2 = gray2_pad[n, 0, i:i + kernel_size, j:j + kernel_size]
                mu = region.mean()
                e_x2 = region2.mean()
                var = max(e_x2 - mu * mu, 0.0)
                sigma = float(np.sqrt(var))
                threshold = mu * (1.0 + k * (sigma / R - 1.0))
                val = 1.0 if gray[n, 0, i, j] > threshold else 0.0
                output[n, :, i, j] = val

    return output


# ---------------------------------------------------------------------------
# 1. 形状テスト
# ---------------------------------------------------------------------------

class TestSauvolaOutputShape:

    def test_single_image(self, session_and_k):
        """単一画像 (N=1) で出力が (1,3,H,W) になること."""
        sess, _ = session_and_k
        img = np.random.rand(1, 3, 32, 32).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (1, 3, 32, 32)

    def test_batch(self, session_and_k):
        """バッチ (N>1) で出力が (N,3,H,W) になること."""
        sess, _ = session_and_k
        img = np.random.rand(2, 3, 32, 32).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (2, 3, 32, 32)

    def test_rectangular(self, session_and_k):
        """非正方形画像で出力形状が入力と同じであること."""
        sess, _ = session_and_k
        img = np.random.rand(1, 3, 24, 48).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (1, 3, 24, 48)


# ---------------------------------------------------------------------------
# 2. 値テスト
# ---------------------------------------------------------------------------

class TestSauvolaValues:

    def test_binary_output(self, session_and_k):
        """出力が 0 または 1 の二値になること."""
        sess, _ = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 32, 32)).astype(np.float32)
        out = _run(sess, img)
        unique_vals = np.unique(out)
        assert set(unique_vals).issubset({0.0, 1.0}), f"非二値の出力: {unique_vals}"

    def test_all_white_image(self, session_and_k):
        """全白画像は閾値が高いため全黒 (背景) になること.

        全白で σ=0 → T = μ = 1.0 → gray (=1.0) > T (=1.0) が False → 出力 0.
        実際には float 精度次第なので 0 または 1 のどちらかになる.
        """
        sess, _ = session_and_k
        img = np.ones((1, 3, 32, 32), dtype=np.float32)
        out = _run(sess, img)
        # 均一なので全ピクセルが同一値になること
        assert np.all(out == out[0, 0, 0, 0]), "均一画像で出力が均一でない"

    def test_all_black_image(self, session_and_k):
        """全黒画像は全て 0 (暗い) になること."""
        sess, _ = session_and_k
        img = np.zeros((1, 3, 32, 32), dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_array_equal(out, 0.0)

    def test_no_nan_inf(self, session_and_k):
        """出力に NaN / Inf が含まれないこと."""
        sess, _ = session_and_k
        rng = np.random.default_rng(7)
        img = rng.random((1, 3, 32, 32)).astype(np.float32)
        out = _run(sess, img)
        assert np.all(np.isfinite(out))

    def test_k_sensitivity(self, session_and_k):
        """k が大きいほど閾値が下がり、白ピクセルが増えること.

        T = μ × [1 + k × (σ/R − 1)] において、
        典型的な画像では σ < R なので (σ/R − 1) < 0。
        k が増えるほど T が下がり、threshold 以上になるピクセルが増える。
        つまり k=1.0 のほうが k=0.0 より白ピクセルが多くなる。
        """
        sess, _ = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 64, 64)).astype(np.float32)
        out_k0 = _run(sess, img, k_val=0.0)
        out_k1 = _run(sess, img, k_val=1.0)
        white_k0 = out_k0.sum()
        white_k1 = out_k1.sum()
        assert white_k1 >= white_k0, "σ < R の典型画像では k が大きいほど白ピクセルが多いはず"

    def test_matches_numpy_reference(self, session_and_k):
        """小画像で NumPy 参照実装と一致すること.

        参照実装が O(H×W×kernel²) なので小画像で検証する.
        """
        sess, k = session_and_k
        rng = np.random.default_rng(42)
        # 最小 kernel より大きく、参照実装が現実的に動く大きさ
        img = rng.random((1, 3, max(k + 4, 20), max(k + 4, 20))).astype(np.float32)
        k_val = 0.5
        expected = _sauvola_numpy(img, k, k_val)
        out = _run(sess, img, k_val)
        np.testing.assert_array_equal(out, expected)

    def test_edge_detection(self, session_and_k):
        """左黒・右白の画像でエッジ付近に白と黒が両方現れること.

        Sauvola は局所適応閾値なので、均一な左半分は全黒に、
        均一な右半分は全白にはならない（局所分散がほぼ0なら閾値≒μ）.
        """
        sess, k = session_and_k
        img = np.zeros((1, 3, 64, 64), dtype=np.float32)
        img[:, :, :, 32:] = 1.0
        out = _run(sess, img)
        # 画像全体が均一黒でも均一白でもないこと
        assert out.min() == 0.0 and out.max() == 1.0, "黒と白の両方が存在するはず"


# ---------------------------------------------------------------------------
# 3. モデル整合性テスト
# ---------------------------------------------------------------------------

class TestSauvolaModelIntegrity:

    @pytest.mark.parametrize("k", KERNEL_SIZES)
    def test_onnx_checker(self, k):
        """onnx.checker がエラーなく通ること."""
        model = onnx.load(str(MODEL_DIR / f"sauvola_{k}x{k}.onnx"))
        onnx.checker.check_model(model)

    @pytest.mark.parametrize("k", KERNEL_SIZES)
    def test_opset_version(self, k):
        """opset version が 17 であること."""
        model = onnx.load(str(MODEL_DIR / f"sauvola_{k}x{k}.onnx"))
        opset = model.opset_import[0].version
        assert opset == 17, f"opset={opset}"

    @pytest.mark.parametrize("k", KERNEL_SIZES)
    def test_io_names(self, k):
        """入出力名が仕様通りであること."""
        model = onnx.load(str(MODEL_DIR / f"sauvola_{k}x{k}.onnx"))
        inputs = [i.name for i in model.graph.input
                  if i.name not in {init.name for init in model.graph.initializer}]
        outputs = [o.name for o in model.graph.output]
        assert inputs == ["input", "k"]
        assert outputs == ["output"]

    @pytest.mark.parametrize("k", KERNEL_SIZES)
    def test_param_meta(self, k):
        """metadata_props に k のパラメータメタデータが埋め込まれていること."""
        model = onnx.load(str(MODEL_DIR / f"sauvola_{k}x{k}.onnx"))
        keys = {p.key for p in model.metadata_props}
        assert "param:k" in keys
