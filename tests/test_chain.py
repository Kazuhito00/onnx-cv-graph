"""ChainOp (直列合成) のテスト.

テスト設計の詳細は TEST_DESIGN.md を参照.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from src.chain import ChainOp
from src.onnx_cv_graph import BinarizeOp, GrayscaleOp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
ELEMENTWISE_DIR = MODEL_DIR / "01_elementwise"

LUMA_WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 3, 1, 1)


@pytest.fixture(scope="module")
def chain_grayscale_binarize():
    """GrayscaleOp → BinarizeOp の ChainOp インスタンスを返す."""
    return ChainOp([GrayscaleOp(), BinarizeOp()])


@pytest.fixture(scope="module")
def model_path(chain_grayscale_binarize):
    """合成モデルをエクスポートしてパスを返す. テスト終了後に削除する."""
    path = MODEL_DIR / "grayscale_binarize.onnx"
    chain_grayscale_binarize.export(path)
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture(scope="module")
def session(model_path):
    """ONNX Runtime の推論セッションを返す."""
    return ort.InferenceSession(str(model_path))


def _run(session, img: np.ndarray, threshold: float) -> np.ndarray:
    """セッションで推論を実行するヘルパー."""
    thr = np.array([threshold], dtype=np.float32)
    return session.run(None, {"input": img, "threshold": thr})[0]


class TestChainOutputShape:
    """出力テンソルの形状を検証するテスト群."""

    def test_single_image(self, session):
        """単一画像 (N=1) で出力が (1,3,H,W) になること."""
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.5)
        assert out.shape == (1, 3, 8, 8)

    def test_batch(self, session):
        """バッチ (N>1) で出力が (N,3,H,W) になること."""
        img = np.random.rand(3, 3, 8, 8).astype(np.float32)
        out = _run(session, img, 0.5)
        assert out.shape == (3, 3, 8, 8)


class TestChainValues:
    """合成結果が単体 op 逐次実行と一致することを検証するテスト群."""

    def test_matches_sequential_execution(self, session):
        """ChainOp の出力が GrayscaleOp → BinarizeOp 逐次実行と一致すること."""
        rng = np.random.default_rng(42)
        img = rng.random((2, 3, 16, 16), dtype=np.float32)
        threshold = 0.4

        # ChainOp の出力
        chain_out = _run(session, img, threshold)

        # 逐次実行: GrayscaleOp → BinarizeOp
        gray_session = ort.InferenceSession(str(ELEMENTWISE_DIR / "grayscale.onnx"))
        gray_out = gray_session.run(None, {"input": img})[0]

        bin_session = ort.InferenceSession(str(ELEMENTWISE_DIR / "binarize.onnx"))
        thr = np.array([threshold], dtype=np.float32)
        sequential_out = bin_session.run(None, {"input": gray_out, "threshold": thr})[0]

        np.testing.assert_allclose(chain_out, sequential_out, atol=1e-5)

    def test_all_white(self, session):
        """全白 → threshold=0.5 で全 1.0 になること."""
        img = np.ones((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img, 0.5)
        np.testing.assert_array_equal(out, np.ones((1, 3, 4, 4), dtype=np.float32))

    def test_all_black(self, session):
        """全黒 → threshold=0.5 で全 0.0 になること."""
        img = np.zeros((1, 3, 4, 4), dtype=np.float32)
        out = _run(session, img, 0.5)
        np.testing.assert_array_equal(out, np.zeros((1, 3, 4, 4), dtype=np.float32))


class TestChainParameterPassthrough:
    """パラメータが正しく透過されることを検証するテスト群."""

    def test_threshold_high(self, session):
        """高い閾値で全 0.0 になること."""
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, img, 0.9)
        np.testing.assert_array_equal(out, np.zeros((1, 3, 4, 4), dtype=np.float32))

    def test_threshold_low(self, session):
        """低い閾値で全 1.0 になること."""
        img = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        out = _run(session, img, 0.1)
        np.testing.assert_array_equal(out, np.ones((1, 3, 4, 4), dtype=np.float32))


class TestChainMetadata:
    """ChainOp のメタデータを検証するテスト群."""

    def test_op_name(self, chain_grayscale_binarize):
        """op_name が "grayscale_binarize" になること."""
        assert chain_grayscale_binarize.op_name == "grayscale_binarize"

    def test_param_meta_merged(self, chain_grayscale_binarize):
        """BinarizeOp の threshold メタデータが引き継がれること."""
        meta = chain_grayscale_binarize.param_meta
        assert "threshold" in meta
        assert meta["threshold"] == (0.0, 1.0, 0.5)

    def test_param_meta_grayscale_empty(self):
        """GrayscaleOp はパラメータを持たないため空であること."""
        assert GrayscaleOp().param_meta == {}

    def test_variants_empty(self):
        """variants() が空リストを返すこと (自動エクスポート対象外)."""
        assert ChainOp.variants() == []

    def test_input_specs(self, chain_grayscale_binarize):
        """input_specs が画像入力 + threshold を含むこと."""
        specs = chain_grayscale_binarize.input_specs
        names = [s[0] for s in specs]
        assert "input" in names
        assert "threshold" in names

    def test_output_specs(self, chain_grayscale_binarize):
        """output_specs が末尾 op の output を返すこと."""
        specs = chain_grayscale_binarize.output_specs
        assert len(specs) == 1
        assert specs[0][0] == "output"
