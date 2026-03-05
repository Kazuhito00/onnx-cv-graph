"""全 OnnxGraphOp 具象サブクラスを models/ にエクスポートする.

各具象クラスの variants() が返す全インスタンスをエクスポートする.
カーネルサイズ等で複数モデルを生成するクラスは variants() をオーバーライドして対応.
グラフ可視化 HTML を assets/ に出力する.
"""

import json
import sys
from pathlib import Path

import onnx
from pyvis.network import Network

# プロジェクトルートを sys.path に追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base import OnnxGraphOp
import src.onnx_cv_graph  # noqa: F401  サブモジュールの import をトリガー
from src.onnx_cv_graph import CATEGORIES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def _all_concrete_subclasses(cls):
    """OnnxGraphOp の全具象サブクラスを再帰的に収集する."""
    result = []
    for sub in cls.__subclasses__():
        if not getattr(sub, "__abstractmethods__", None):
            result.append(sub)
        result.extend(_all_concrete_subclasses(sub))
    return result


def _shape_str(value_info) -> str:
    """TensorType の shape を文字列化する."""
    t = value_info.type.tensor_type
    if not t.shape.dim:
        return ""
    dims = []
    for d in t.shape.dim:
        if d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append(str(d.dim_value))
    return "×".join(dims)


def _export_graph_html(onnx_path: Path, html_path: Path) -> None:
    """ONNX モデルを pyvis でインタラクティブ HTML に可視化する."""
    model = onnx.load(str(onnx_path))
    graph = model.graph

    net = Network(
        directed=True,
        height="400px",
        width="100%",
        bgcolor="#ffffff",
    )
    net.set_options("""
    {
      "layout": { "hierarchical": { "enabled": true, "direction": "UD", "sortMethod": "directed", "nodeSpacing": 120, "levelSeparation": 60 } },
      "physics": { "enabled": false },
      "edges": { "arrows": { "to": { "enabled": true } }, "color": { "color": "#e2e8f0", "highlight": "#c7d2fe" }, "smooth": { "type": "cubicBezier" } }
    }
    """)

    init_names = {init.name for init in graph.initializer}

    # 安定したノード ID を生成するため、インデックスベースの ID を使用
    node_ids = {}
    for idx, node in enumerate(graph.node):
        node_ids[idx] = node.name or f"{node.op_type}_{idx}"

    # 入力ノード (initializer は除外)
    for inp in graph.input:
        if inp.name in init_names:
            continue
        label = f"{inp.name}\n{_shape_str(inp)}"
        net.add_node(inp.name, label=label, color="#a7f3d0", shape="box",
                     font={"size": 12, "color": "#334155"},
                     borderWidth=0)

    # 演算ノード
    for idx, node in enumerate(graph.node):
        net.add_node(node_ids[idx], label=node.op_type, color="#c7d2fe",
                     shape="box", font={"size": 14, "bold": True, "color": "#334155"},
                     borderWidth=0)

    # 出力ノード
    for out in graph.output:
        label = f"{out.name}\n{_shape_str(out)}"
        net.add_node(out.name, label=label, color="#fed7aa", shape="box",
                     font={"size": 12, "color": "#334155"},
                     borderWidth=0)

    # テンソル名 → 生成元ノード ID のマッピング
    producer = {}
    for inp in graph.input:
        if inp.name not in init_names:
            producer[inp.name] = inp.name
    for idx, node in enumerate(graph.node):
        for o in node.output:
            producer[o] = node_ids[idx]

    # エッジ接続 (入力元 → 演算ノード)
    for idx, node in enumerate(graph.node):
        for i in node.input:
            if i in producer:
                net.add_edge(producer[i], node_ids[idx], title=i)

    # 演算ノード → グラフ出力
    output_names = {out.name for out in graph.output}
    for idx, node in enumerate(graph.node):
        for o in node.output:
            if o in output_names:
                net.add_edge(node_ids[idx], o, title=o)

    net.save_graph(str(html_path))

    # pyvis テンプレートの余白・不要要素を除去してコンパクト化
    html = html_path.read_text(encoding="utf-8")
    html = html.replace('<center>\n<h1></h1>\n</center>', '')
    html = html.replace('<center>\n          <h1></h1>\n        </center>', '')
    html = html.replace('border: 1px solid lightgray;', 'border: none;')
    # body/card のマージン・パディングを除去
    html = html.replace('<body>', '<body style="margin:0;padding:0;overflow:hidden;">')
    html = html.replace('<div class="card" style="width: 100%">',
                        '<div style="width:100%;margin:0;padding:0;">')
    html = html.replace('class="card-body"', '')
    html_path.write_text(html, encoding="utf-8")


def main():
    op_classes = _all_concrete_subclasses(OnnxGraphOp)
    if not op_classes:
        print("具象 OnnxGraphOp サブクラスが見つかりません.")
        return

    # op_name → カテゴリ ID の逆引きマップを構築
    op_to_category = {}
    for cat in CATEGORIES:
        for op_cls in cat["ops"]:
            for op in op_cls.variants():
                op_to_category[op.op_name] = cat["id"]

    params = {}
    for op_cls in op_classes:
        for op in op_cls.variants():
            cat_id = op_to_category.get(op.op_name, "")
            out_dir = MODELS_DIR / cat_id if cat_id else MODELS_DIR
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{op.op_name}.onnx"
            op.export(out_path)
            print(f"エクスポート完了: {out_path}")
            params[op.op_name] = {k: list(v) for k, v in op.param_meta.items()}

    # カテゴリ情報を構築
    categories = []
    for cat in CATEGORIES:
        model_names = []
        for op_cls in cat["ops"]:
            for op in op_cls.variants():
                model_names.append(op.op_name)
        cat_entry = {
            "id": cat["id"],
            "label_ja": cat["label_ja"],
            "label_en": cat["label_en"],
            "models": model_names,
        }
        if cat.get("hidden"):
            cat_entry["hidden"] = True
        categories.append(cat_entry)

    meta = {
        "categories": categories,
        "params": params,
    }

    # パラメータメタデータを JSON に書き出す (web_demo 用)
    meta_path = MODELS_DIR / "models_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"メタデータ出力完了: {meta_path}")

    # グラフ可視化 HTML を assets/ にカテゴリ別サブディレクトリで出力
    for onnx_path in sorted(MODELS_DIR.glob("*/*.onnx")):
        cat_id = onnx_path.parent.name
        html_dir = ASSETS_DIR / cat_id
        html_dir.mkdir(parents=True, exist_ok=True)
        html_path = html_dir / f"{onnx_path.stem}_graph.html"
        _export_graph_html(onnx_path, html_path)
        print(f"グラフ可視化出力完了: {html_path}")


if __name__ == "__main__":
    main()
