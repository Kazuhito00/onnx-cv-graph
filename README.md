# onnx-cv-prototype

ONNX オペレーションのみで画像処理を行うツールキット。

Python 側は ONNX グラフの構築・エクスポートのみを担い、推論時の計算はすべて ONNX Runtime 内で完結する。

## 設計思想

- **ONNX-only** — 推論時に Python の画像処理ライブラリに依存しない
- **グラフ構築と推論の分離** — `OnnxGraphOp` ベースクラスでグラフ定義を抽象化し、エクスポート済み `.onnx` ファイルを任意のランタイムで実行可能
- **チェーン合成** — `ChainOp` により複数オペレーションを直列結合し、1つの ONNX モデルとしてエクスポート可能
- **[0, 1] float32 RGB 統一** — 入出力テンソルの値域・形式を統一し、オペレーション間の接続を再正規化なしで行えるようにする
- **パイプライン一体化** — 前処理・推論・後処理を単一の ONNX グラフに封入し、処理ロジック・閾値・正規化係数をすべて `.onnx` ファイル1つで配布できる。環境間の前処理差異によるバグを排除する
- **クロスプラットフォーム再現性** — 計算グラフが固定されるため、Python / Native / Web / モバイル / 組み込みのどの ONNX Runtime でも同一入力に対して同一出力が保証される
- **GPU 自動オフロード** — 画像処理がテンソル演算に落ちているため、ONNX Runtime の WebGPU / CUDA / DirectML バックエンドにより Conv・Mul・Add・Resize 等が自動的に GPU 並列実行される
- **学習化への拡張性** — 固定カーネルのフィルタや正規化係数を将来的に学習可能なパラメータに置換する際、ONNX パイプライン上でそのまま拡張できる

## ファイル構成

```
CLAUDE.md                # Claude Code 向けガイド
MODELS.md                # 実現可能な画像処理の一覧と実装状況
TEST_DESIGN.md           # テスト設計ガイド
requirements.txt         # Python 依存パッケージ
assets/                  # サンプル画像・グラフ可視化 HTML
sample_app.py            # Streamlit デモアプリ
sample_app.html          # onnxruntime-web ブラウザデモ
src/
  base.py                # OnnxGraphOp 抽象ベースクラス
  chain.py               # ChainOp — 複数 op の直列合成
  export_all.py          # 全モデル自動エクスポート + models_meta.json 生成
  onnx_cv_graph/         # 各画像処理オペレーション (OnnxGraphOp サブクラス)
models/                  # エクスポート先 (*.onnx + models_meta.json, 自動生成)
tests/                   # pytest テスト (test_{op_name}.py)
```

## セットアップ

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 使い方

### モデルのエクスポート

```bash
python src/export_all.py
```

`models/` に全オペレーションの `.onnx` ファイルが生成される。

### 推論の実行例

```python
import cv2
import numpy as np
import onnxruntime as ort

# 画像読み込み → NCHW float32 RGB [0,1] に変換
bgr = cv2.imread("assets/sample.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
img = rgb.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0  # (1, 3, H, W)

# 推論
session = ort.InferenceSession("models/grayscale.onnx")
gray = session.run(None, {"input": img})[0]  # (1, 3, H, W)

# 表示用に戻す
out = (gray[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)  # HWC uint8
cv2.imshow("result", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
```

### チェーン合成 (ChainOp)

複数のオペレーションを直列に合成し、1つの ONNX モデルとしてエクスポートできる。

```python
from src.chain import ChainOp
from src.onnx_cv_graph import GrayscaleOp, BinarizeOp

# 合成してエクスポート
chain = ChainOp([GrayscaleOp(), BinarizeOp()])
chain.export("models/grayscale_binarize.onnx")
```

- 各 op のパラメータ入力 (threshold 等) はそのままグラフ入力に昇格する
- パラメータ名が衝突する場合は `{op_name}.{param}` にリネームされる
- `param_meta` は全 op からマージされ、`metadata_props` に埋め込まれる
- `export_all.py` の自動エクスポート対象外 (`variants()` が空リストを返す)

## 実装済みオペレーション

実現可能な画像処理の全一覧と実装状況は [MODELS.md](MODELS.md) を参照。

## テスト

```bash
pytest tests/ -v
```

テストは以下の4カテゴリで構成される（詳細は [TEST_DESIGN.md](TEST_DESIGN.md) を参照）:

1. **形状テスト** — 出力テンソルの shape が仕様どおりであること
2. **値テスト** — NumPy 参照実装との数値一致 (`atol=1e-5`)
3. **OpenCV 比較テスト** — `cv2.cvtColor` との比較で uint8 ±1 以内
4. **モデル整合性テスト** — ONNX checker 通過、opset version 確認（任意）

## テンソル規約

全オペレーション共通のルール。チェーン合成時に再正規化なしで接続できることを保証する。

| 項目 | 規約 |
|------|------|
| レイアウト | NCHW (`N`: バッチ, `C`: チャネル, `H`: 高さ, `W`: 幅) |
| データ型 | float32 |
| チャネル順 | RGB |
| 値域 | **[0, 1]** (入力・出力とも) |

- 新しいオペレーションを実装する際は、入力・出力ともにこの規約に従うこと
- 出力が [0, 1] を超えうる演算 (エッジ検出、シャープ化等) では **末尾に Clip(0, 1) ノードを必ず入れる**
- [0, 255] uint8 画像を扱う場合は、モデル外で `/ 255.0` による正規化を行う
- グレースケールや2値化の出力も 3ch 同一値で統一し、チェーン合成に対応する

## 2入力オペレーション規約

ブレンド等、画像2枚を入力にとるオペレーションでは以下の命名規則に従う:

| 入力名 | 役割 | 例 |
|--------|------|------|
| `input` | 主画像 `(N,3,H,W)` float32 | ベース画像 |
| `input2` | 副画像 `(N,3,H,W)` float32 | ブレンド対象 |

- 両入力とも同じ shape であること
- `sample_app.py` では `input2` が検出された場合、2枚目の画像読み込み UI を表示する

## パラメータメタデータ規約

推論時に外部から渡すパラメータ（閾値など）がある場合、ONNX モデルの `metadata_props` に値域とデフォルト値を埋め込む。

### 形式

| キー | 値 | 例 |
|------|------|------|
| `param:{入力名}` | `min,max,default` | `param:threshold` → `0.0,1.0,0.5` |

### 実装方法

サブクラスで `param_meta` プロパティをオーバーライドする。`build_model()` が自動的に `metadata_props` へ埋め込む。

```python
@property
def param_meta(self) -> Dict[str, ParamMeta]:
    return {"threshold": (0.0, 1.0, 0.5)}  # (min, max, default)
```

### アプリ側での利用

```python
import onnx
model = onnx.load("binarize.onnx")
for prop in model.metadata_props:
    if prop.key.startswith("param:"):
        name = prop.key[len("param:"):]
        lo, hi, default = (float(v) for v in prop.value.split(","))
```

`sample_app.py` はこのメタデータを読み取り、スライダーの範囲・デフォルト値を自動設定する。メタデータが無いパラメータにはフォールバック `[0.0, 1.0, 0.5]` を使用する。

## Web デモ (onnxruntime-web)

ブラウザ内で ONNX モデルの推論を実行するデモページ。サーバーサイド不要で動作する。

```bash
# モデルとメタデータを生成
python src/export_all.py

# ローカルサーバー起動
python -m http.server 8080

# ブラウザで開く
# http://localhost:8080/sample_app.html
```

- `sample_app.html` — onnxruntime-web (CDN) を使用した単一 HTML デモ
- `models/models_meta.json` — `export_all.py` が自動生成するパラメータメタデータ (スライダーの min/max/default)
- `assets/sample.jpg` をデフォルト画像として読み込み、ファイルアップロードにも対応

## 技術詳細

- **opset version**: 17 (ONNX Runtime 1.14+ で広くサポート)
- **IR version**: 8


## 新しいオペレーションの追加方法

1. `src/onnx_cv_graph/` に `OnnxGraphOp` のサブクラスを作成
2. `op_name`, `input_specs`, `output_specs`, `build_graph` を実装
3. **入力・出力ともに [0, 1] float32 RGB (NCHW) のテンソル規約に従うこと**
4. **出力が [0, 1] を超えうる場合は末尾に Clip(0, 1) ノードを入れること**
5. 推論時パラメータがある場合は `param_meta` をオーバーライド（[パラメータメタデータ規約](#パラメータメタデータ規約) を参照）
6. カーネルサイズ等で複数モデルを生成する場合は `variants()` をオーバーライド
7. 2入力オペレーションは `"input"` / `"input2"` の命名規則に従う（[2入力オペレーション規約](#2入力オペレーション規約) を参照）
8. `src/onnx_cv_graph/__init__.py` に import を追加
9. `tests/test_{op_name}.py` を作成（[テスト設計ガイド](TEST_DESIGN.md) に従う）
10. `python src/export_all.py` && `pytest tests/ -v` で検証
