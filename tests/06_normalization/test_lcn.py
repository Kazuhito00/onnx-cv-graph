"""局所コントラスト正規化 (LCN) モデルのテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
15×15 / 31×31 / 63×63 の全バリアントをテストする.

式: output = Sigmoid((input − local_mean) / (local_std + ε))
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

KERNEL_SIZES = [15, 31]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    missing = any(
        not (MODEL_DIR / f"lcn_{k}x{k}.onnx").exists() for k in KERNEL_SIZES
    )
    if missing:
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )
    for k in KERNEL_SIZES:
        p = MODEL_DIR / f"lcn_{k}x{k}.onnx"
        assert p.exists(), f"モデルが見つかりません: {p}"


@pytest.fixture(
    scope="module",
    params=KERNEL_SIZES,
    ids=[f"sess_{k}x{k}" for k in KERNEL_SIZES],
)
def session_and_k(request):
    """(InferenceSession, kernel_size) のタプルを返す."""
    k = request.param
    sess = ort.InferenceSession(str(MODEL_DIR / f"lcn_{k}x{k}.onnx"))
    return sess, k


# ---------------------------------------------------------------------------
# Helper: 推論実行
# ---------------------------------------------------------------------------

def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


# ---------------------------------------------------------------------------
# Helper: NumPy 参照実装
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def _lcn_numpy(img: np.ndarray, kernel_size: int, eps: float = 1e-5) -> np.ndarray:
    """LCN の NumPy 参照実装.

    img: (N, 3, H, W) float32
    AveragePool(reflect pad) による局所平均・分散を計算し Sigmoid で写像する.
    """
    img = img.astype(np.float32)
    N, C, H, W = img.shape
    pad = kernel_size // 2
    output = np.zeros_like(img)

    for c in range(C):
        ch = img[:, c:c+1, :, :]  # (N,1,H,W)
        ch_pad = np.pad(ch, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")
        ch2_pad = np.pad(ch ** 2, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")

        for n in range(N):
            for i in range(H):
                for j in range(W):
                    region = ch_pad[n, 0, i:i + kernel_size, j:j + kernel_size]
                    region2 = ch2_pad[n, 0, i:i + kernel_size, j:j + kernel_size]
                    mu = region.mean()
                    e_x2 = region2.mean()
                    var = max(float(e_x2 - mu * mu), 0.0)
                    sigma = float(np.sqrt(var))
                    normalized = (float(ch[n, 0, i, j]) - mu) / (sigma + eps)
                    output[n, c, i, j] = float(_sigmoid(np.array(normalized)))

    return output


# ---------------------------------------------------------------------------
# 1. 形状テスト
# ---------------------------------------------------------------------------

class TestLcnOutputShape:

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

class TestLcnValues:

    def test_output_range(self, session_and_k):
        """Sigmoid 出力なので (0, 1) 範囲内であること."""
        sess, _ = session_and_k
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 32, 32)).astype(np.float32)
        out = _run(sess, img)
        assert out.min() > 0.0 - 1e-6
        assert out.max() < 1.0 + 1e-6

    def test_no_nan_inf(self, session_and_k):
        """出力に NaN / Inf が含まれないこと."""
        sess, _ = session_and_k
        rng = np.random.default_rng(7)
        img = rng.random((1, 3, 32, 32)).astype(np.float32)
        out = _run(sess, img)
        assert np.all(np.isfinite(out))

    def test_uniform_image_is_midgray(self, session_and_k):
        """均一画像は σ=0 → normalized=0 → Sigmoid(0)=0.5 になること.

        ONNX reflect pad の制約 (pad_before + pad_after ≤ d-1) を満たすため、
        画像サイズはカーネルサイズより十分大きくする必要がある。
        """
        sess, k = session_and_k
        # reflect pad 制約: pad*2 <= H-1 → H >= k+1。余裕を持って k*2+2 を使う。
        size = max(k * 2 + 2, 64)
        img = np.full((1, 3, size, size), 0.6, dtype=np.float32)
        out = _run(sess, img)
        # Sigmoid(0) = 0.5。float32 精度の累積誤差を考慮して atol=2e-3
        np.testing.assert_allclose(out, 0.5, atol=2e-3)

    def test_bright_above_mean_is_over_half(self, session_and_k):
        """局所平均より明るいピクセルは 0.5 より大きくなること."""
        sess, k = session_and_k
        # 画像の中央だけ明るくする (境界影響を避けるため十分大きく)
        size = max(k * 2 + 4, 80)
        img = np.full((1, 3, size, size), 0.3, dtype=np.float32)
        cx, cy = size // 2, size // 2
        img[:, :, cy, cx] = 0.9  # 中央を明るく
        out = _run(sess, img)
        assert out[0, :, cy, cx].mean() > 0.5, "局所平均より明るい点は 0.5 超のはず"

    def test_dark_below_mean_is_under_half(self, session_and_k):
        """局所平均より暗いピクセルは 0.5 より小さくなること."""
        sess, k = session_and_k
        size = max(k * 2 + 4, 80)
        img = np.full((1, 3, size, size), 0.7, dtype=np.float32)
        cx, cy = size // 2, size // 2
        img[:, :, cy, cx] = 0.1  # 中央を暗く (テキスト想定)
        out = _run(sess, img)
        assert out[0, :, cy, cx].mean() < 0.5, "局所平均より暗い点は 0.5 未満のはず"

    def test_matches_numpy_reference(self, session_and_k):
        """小画像で NumPy 参照実装と一致すること (atol=1e-4)."""
        sess, k = session_and_k
        rng = np.random.default_rng(42)
        size = max(k + 4, 20)
        img = rng.random((1, 3, size, size)).astype(np.float32)
        expected = _lcn_numpy(img, k)
        out = _run(sess, img)
        np.testing.assert_allclose(out, expected, atol=1e-4)

    def test_contrast_enhancement(self, session_and_k):
        """低コントラスト領域でコントラストが強調されること.

        局所的に似た値のグラデーションに LCN を適用すると、
        微細な差が拡大されて出力の分散が増加することを確認する.
        """
        sess, k = session_and_k
        size = max(k * 2 + 4, 80)
        # 緩やかなグラデーション (低コントラスト)
        ramp = np.linspace(0.4, 0.6, size, dtype=np.float32)
        img = np.tile(ramp, (1, 3, size, 1))  # 水平グラデーション
        out = _run(sess, img)
        # 出力の標準偏差が入力より大きくなる (コントラスト強調)
        assert out.std() >= img.std() - 1e-4, "LCN でコントラストが強調されるはず"


# ---------------------------------------------------------------------------
# 3. モデル整合性テスト
# ---------------------------------------------------------------------------

class TestLcnModelIntegrity:

    @pytest.mark.parametrize("k", KERNEL_SIZES)
    def test_onnx_checker(self, k):
        """onnx.checker がエラーなく通ること."""
        model = onnx.load(str(MODEL_DIR / f"lcn_{k}x{k}.onnx"))
        onnx.checker.check_model(model)

    @pytest.mark.parametrize("k", KERNEL_SIZES)
    def test_opset_version(self, k):
        """opset version が 17 であること."""
        model = onnx.load(str(MODEL_DIR / f"lcn_{k}x{k}.onnx"))
        opset = model.opset_import[0].version
        assert opset == 17, f"opset={opset}"

    @pytest.mark.parametrize("k", KERNEL_SIZES)
    def test_io_names(self, k):
        """入出力名が仕様通りであること."""
        model = onnx.load(str(MODEL_DIR / f"lcn_{k}x{k}.onnx"))
        inputs = [i.name for i in model.graph.input
                  if i.name not in {init.name for init in model.graph.initializer}]
        outputs = [o.name for o in model.graph.output]
        assert inputs == ["input"]
        assert outputs == ["output"]
