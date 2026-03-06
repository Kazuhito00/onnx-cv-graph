"""ONNX グラフ操作の抽象ベースクラス."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import onnx
from onnx import GraphProto, ModelProto, StringStringEntryProto, TensorProto, checker

OPSET_VERSION = 17

TensorSpec = Tuple[str, int, List[str | int | None]]
"""テンソル仕様 (名前, 要素型, 形状). 形状要素は str(動的次元), int, None のいずれか."""

ParamMeta = Tuple[float, float, float]
"""パラメータメタデータ (min, max, default). モデルの metadata_props に埋め込まれる."""


class OnnxGraphOp(ABC):
    """ONNX グラフ操作の抽象基底クラス.

    サブクラスで op_name / input_specs / output_specs / build_graph を実装する.
    build_model / export は共通ロジックとして提供.

    テンソル規約:
        - 画像テンソルは NCHW レイアウト, float32, RGB チャネル順
        - 標準画像 op の値域は [0, 1]. 超えうる演算では末尾に Clip(0, 1) を入れること.

    パラメータメタデータ:
        - 推論時パラメータがある場合 param_meta で (min, max, default) を定義する
        - build_model() が metadata_props に "param:{name}" = "min,max,default" 形式で埋め込む
        - アプリ側はこのメタデータを読み取り UI コントロールを自動生成できる

    バリアント生成:
        - カーネルサイズ等でグラフ構造が変わる場合は variants() をオーバーライドする
        - export_all.py は variants() が返す全インスタンスをエクスポートする
        - デフォルトでは自身1つのみを返す

    2入力オペレーション:
        - 画像2枚を入力にとる場合 input_specs で "input" と "input2" を定義する
        - "input" が主画像、"input2" が副画像 (ブレンド対象等)
    """

    @property
    @abstractmethod
    def op_name(self) -> str:
        """操作名 (エクスポート時のファイル名に使用)."""
        ...

    @property
    @abstractmethod
    def input_specs(self) -> List[TensorSpec]:
        """入力テンソルの仕様リスト. チェーン合成時の互換性検証に使用."""
        ...

    @property
    @abstractmethod
    def output_specs(self) -> List[TensorSpec]:
        """出力テンソルの仕様リスト. チェーン合成時の互換性検証に使用."""
        ...

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        """推論時パラメータのメタデータ. キーは入力名, 値は (min, max, default).

        パラメータを持たないオペレーションはオーバーライド不要 (空辞書を返す).
        """
        return {}

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        """エクスポート対象のインスタンスリストを返す.

        カーネルサイズ等で複数モデルを生成する場合にオーバーライドする.
        デフォルトではデフォルトコンストラクタで1インスタンスを返す.
        """
        return [cls()]

    @abstractmethod
    def build_graph(self) -> GraphProto:
        """ONNX GraphProto を構築して返す."""
        ...

    def build_model(self) -> ModelProto:
        """グラフから ModelProto を生成し、checker で検証する.

        param_meta が定義されていれば metadata_props に埋め込む.
        """
        graph = self.build_graph()
        model = onnx.helper.make_model(graph, opset_imports=[
            onnx.helper.make_opsetid("", OPSET_VERSION),
        ])
        model.ir_version = 8
        # パラメータメタデータを metadata_props に埋め込む
        for name, (lo, hi, default) in self.param_meta.items():
            model.metadata_props.append(
                StringStringEntryProto(key=f"param:{name}", value=f"{lo},{hi},{default}")
            )
        checker.check_model(model)
        return model

    def export(self, path: str | Path) -> Path:
        """モデルを指定パスに .onnx ファイルとして保存する."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = self.build_model()
        onnx.save(model, str(path))
        return path
