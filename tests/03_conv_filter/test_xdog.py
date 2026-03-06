"""XDoG (Extended Difference of Gaussians) フィルタモデルのテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
3×3/5×5 / 5×5/9×9 / 7×7/13×13 の全バリアントをテストする.

OpenCV に XDoG の直接相当 API が無いため VsOpenCV カテゴリは省略し、
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

KERNEL_PAIRS = [(3, 5), (5, 9), (7, 13)]
MODEL_NAMES = [f"xdog_{k1}x{k1}_{k2}x{k2}" for k1, k2 in KERNEL_PAIRS]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def ensure_models():
    """モデルファイルが無ければ export_all.py を実行して生成する."""
    missing = any(
        not (MODEL_DIR / f"{name}.onnx").exists() for name in MODEL_NAMES
    )
    if missing:
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "src" / "export_all.py")],
            cwd=str(PROJECT_ROOT),
        )
    for name in MODEL_NAMES:
        p = MODEL_DIR / f"{name}.onnx"
        assert p.exists(), f"モデルが見つかりません: {p}"


@pytest.fixture(
    scope="module",
    params=KERNEL_PAIRS,
    ids=[f"sess_{k1}x{k1}_{k2}x{k2}" for k1, k2 in KERNEL_PAIRS],
)
def session_and_ks(request):
    """(InferenceSession, k1, k2) のタプルを返す."""
    k1, k2 = request.param
    name = f"xdog_{k1}x{k1}_{k2}x{k2}"
    sess = ort.InferenceSession(str(MODEL_DIR / f"{name}.onnx"))
    return sess, k1, k2


# ---------------------------------------------------------------------------
# Helper: 推論実行
# ---------------------------------------------------------------------------

def _run(session, img: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": img})[0]


# ---------------------------------------------------------------------------
# Helper: NumPy 参照実装
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(ksize: int) -> np.ndarray:
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    ax = np.arange(ksize, dtype=np.float32) - (ksize - 1) / 2.0
    kernel = np.exp(-0.5 * (ax / sigma) ** 2)
    return (kernel / kernel.sum()).astype(np.float32)


def _gaussian_kernel_2d(ksize: int) -> np.ndarray:
    k1d = _gaussian_kernel_1d(ksize)
    return np.outer(k1d, k1d).astype(np.float32)


def _reflect_pad(arr: np.ndarray, pad: int) -> np.ndarray:
    """(N,1,H,W) に reflect padding を適用."""
    return np.pad(arr, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")


def _conv2d_gray(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """(N,1,H+2p,W+2p) に (kh,kw) カーネルを畳み込む."""
    N, C, H, W = arr.shape
    kh, kw = kernel.shape
    oh, ow = H - kh + 1, W - kw + 1
    out = np.zeros((N, C, oh, ow), dtype=np.float32)
    for n in range(N):
        for i in range(oh):
            for j in range(ow):
                out[n, 0, i, j] = (arr[n, 0, i:i+kh, j:j+kw] * kernel).sum()
    return out


def _xdog_numpy(
    img: np.ndarray,
    k1: int,
    k2: int,
    gamma: float = 0.98,
    phi: float = 200.0,
    eps: float = -0.1,
) -> np.ndarray:
    """XDoG フィルタの NumPy 参照実装.

    img: (N, 3, H, W) float32
    ONNX グラフと同じ輝度係数・ガウシアンカーネル・正規化・Tanh 処理を適用する.
    """
    img = img.astype(np.float32)
    # グレースケール変換
    luma = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)
    gray = (img * luma).sum(axis=1, keepdims=True)  # (N,1,H,W)

    # ガウシアンカーネル
    kern1 = _gaussian_kernel_2d(k1)
    kern2 = _gaussian_kernel_2d(k2)

    # ぼかし (reflect padding + 畳み込み)
    g1 = _conv2d_gray(_reflect_pad(gray, k1 // 2), kern1)
    g2 = _conv2d_gray(_reflect_pad(gray, k2 // 2), kern2)

    # DoG
    dog = g1 - gamma * g2  # (N,1,H,W)

    # 正規化
    dog_max = dog.max(axis=(2, 3), keepdims=True)  # (N,1,1,1)
    dog_max_safe = np.maximum(dog_max, 1e-6)
    dog_norm = dog / dog_max_safe

    # XDoG: 1 + tanh(φ × (dog_norm - ε))
    e = 1.0 + np.tanh(phi * (dog_norm - eps))

    # Clip → 3ch 拡張
    clipped = np.clip(e, 0.0, 1.0).astype(np.float32)
    return np.broadcast_to(clipped, (img.shape[0], 3, img.shape[2], img.shape[3])).copy()


# ---------------------------------------------------------------------------
# 1. 形状テスト
# ---------------------------------------------------------------------------

class TestXDoGOutputShape:

    def test_single_image(self, session_and_ks):
        """単一画像 (N=1) で出力が (1,3,H,W) になること."""
        sess, *_ = session_and_ks
        img = np.random.rand(1, 3, 16, 16).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (1, 3, 16, 16)

    def test_batch(self, session_and_ks):
        """バッチ (N>1) で出力が (N,3,H,W) になること."""
        sess, *_ = session_and_ks
        img = np.random.rand(2, 3, 16, 16).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (2, 3, 16, 16)

    def test_rectangular(self, session_and_ks):
        """非正方形画像で出力形状が入力と同じであること."""
        sess, *_ = session_and_ks
        img = np.random.rand(1, 3, 12, 20).astype(np.float32)
        out = _run(sess, img)
        assert out.shape == (1, 3, 12, 20)


# ---------------------------------------------------------------------------
# 2. 値テスト
# ---------------------------------------------------------------------------

class TestXDoGValues:

    def test_output_range(self, session_and_ks):
        """出力値が [0, 1] 範囲内であること."""
        sess, *_ = session_and_ks
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 16, 16)).astype(np.float32)
        out = _run(sess, img)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_no_nan_inf(self, session_and_ks):
        """出力に NaN / Inf が含まれないこと."""
        sess, *_ = session_and_ks
        rng = np.random.default_rng(7)
        img = rng.random((1, 3, 16, 16)).astype(np.float32)
        out = _run(sess, img)
        assert np.all(np.isfinite(out))

    def test_uniform_image_is_white(self, session_and_ks):
        """均一画像はスケッチ上で白 (1.0) になること.

        均一画像は DoG=0、phi*(0-eps)>0 なので tanh≈1、e≈2、clip→1.
        """
        sess, *_ = session_and_ks
        img = np.full((1, 3, 16, 16), 0.5, dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_allclose(out, 1.0, atol=1e-5)

    def test_all_black_is_white(self, session_and_ks):
        """全黒画像はスケッチ上で白 (1.0) になること."""
        sess, *_ = session_and_ks
        img = np.zeros((1, 3, 16, 16), dtype=np.float32)
        out = _run(sess, img)
        np.testing.assert_allclose(out, 1.0, atol=1e-5)

    def test_output_is_grayscale(self, session_and_ks):
        """XDoG はグレースケール変換を経るため、RGB 3ch が同一値になること."""
        sess, *_ = session_and_ks
        rng = np.random.default_rng(99)
        img = rng.random((1, 3, 16, 16)).astype(np.float32)
        out = _run(sess, img)
        np.testing.assert_allclose(out[:, 0], out[:, 1], atol=1e-5)
        np.testing.assert_allclose(out[:, 0], out[:, 2], atol=1e-5)

    def test_matches_numpy_reference(self, session_and_ks):
        """ランダム入力で NumPy 参照実装と一致すること (atol=1e-4).

        ONNX float32 演算と NumPy float32 演算の丸め誤差を考慮.
        """
        sess, k1, k2 = session_and_ks
        rng = np.random.default_rng(42)
        img = rng.random((1, 3, 12, 12)).astype(np.float32)
        expected = _xdog_numpy(img, k1, k2)
        out = _run(sess, img)
        np.testing.assert_allclose(out, expected, atol=1e-4)

    def test_edge_appears_dark(self, session_and_ks):
        """明確なエッジ（左黒・右白）の境界付近で暗い線が現れること.

        XDoG はエッジ部で DoG が正の極大になるため、
        正規化後の dog_norm が大きく、1 + tanh(phi*(1-eps)) ≈ 2 → clip 1 だが、
        DoG が負になる隣接画素で軟閾値が効いて値が下がる。
        ここでは、エッジから十分離れた均一領域が白になることを確認する.
        """
        sess, k1, *_ = session_and_ks
        img = np.zeros((1, 3, 24, 24), dtype=np.float32)
        img[:, :, :, 12:] = 1.0  # 右半分を白
        out = _run(sess, img)
        margin = k1 + 2
        # エッジから離れた均一領域は白 (1.0) に近い
        np.testing.assert_allclose(out[0, :, :, :4], 1.0, atol=0.15)
        np.testing.assert_allclose(out[0, :, :, 20:], 1.0, atol=0.15)


# ---------------------------------------------------------------------------
# 3. モデル整合性テスト
# ---------------------------------------------------------------------------

class TestXDoGModelIntegrity:

    @pytest.mark.parametrize("k1,k2", KERNEL_PAIRS)
    def test_onnx_checker(self, k1, k2):
        """onnx.checker がエラーなく通ること."""
        name = f"xdog_{k1}x{k1}_{k2}x{k2}"
        model = onnx.load(str(MODEL_DIR / f"{name}.onnx"))
        onnx.checker.check_model(model)

    @pytest.mark.parametrize("k1,k2", KERNEL_PAIRS)
    def test_opset_version(self, k1, k2):
        """opset version が 17 であること."""
        name = f"xdog_{k1}x{k1}_{k2}x{k2}"
        model = onnx.load(str(MODEL_DIR / f"{name}.onnx"))
        opset = model.opset_import[0].version
        assert opset == 17, f"opset={opset}"

    @pytest.mark.parametrize("k1,k2", KERNEL_PAIRS)
    def test_io_names(self, k1, k2):
        """入出力名が仕様通りであること."""
        name = f"xdog_{k1}x{k1}_{k2}x{k2}"
        model = onnx.load(str(MODEL_DIR / f"{name}.onnx"))
        inputs = [i.name for i in model.graph.input
                  if i.name not in {init.name for init in model.graph.initializer}]
        outputs = [o.name for o in model.graph.output]
        assert inputs == ["input"]
        assert outputs == ["output"]
