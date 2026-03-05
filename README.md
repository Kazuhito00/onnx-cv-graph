# onnx-cv-graph

ONNX グラフだけで画像処理モデルを組み立てるための実験用リポジトリです。

このリポジトリでは、Python はモデル生成専用です。実行時の推論処理は、エクスポートした ONNX モデルを ONNX Runtime で実行します。

## 何をするプロジェクトか

- 画像処理オペレーションを `OnnxGraphOp` として実装する
- 各オペレーションを `.onnx` として書き出す
- 必要なら `ChainOp` で複数オペレーションを直列にまとめる
- Python デモ / Web デモで同じモデルを動かす

## 設計思想

- 推論実行系は ONNX Runtime に寄せ、Python 実装との差分で挙動が変わらないようにする
- 前処理・推論・後処理を可能な範囲で単一 ONNX グラフにまとめ、配布単位をモデル1つに寄せる
- 入出力テンソル規約を固定し、オペレーション同士を再正規化なしで接続できる状態を保つ
- `ChainOp` で処理を直列合成し、実行時オーケストレーションをモデル側に寄せる
- パラメータは `metadata_props` に記録し、実行側 UI やツールがモデル単体から設定情報を復元できるようにする
- 同一モデルを Python / Web / ネイティブで再利用し、環境差による前後処理バグを減らす
- テンソル演算として表現できる処理は、CUDA / DirectML / WebGPU などの実行プロバイダに委譲しやすくする

## 設計上の前提

- 推論時は ONNX Runtime を使う
- テンソルは `NCHW / RGB / float32 / [0,1]` に統一する
- モデル間をつなぐときも同じ規約を守る

## ファイル構成

```text
README.md                # README
MODELS.md                # 実装済み・実装候補オペレーション一覧
TEST_DESIGN.md           # テスト方針
requirements.txt         # Python 依存パッケージ
sample_app.html          # onnxruntime-web デモ
assets/                  # サンプル画像・グラフ可視化 HTML
src/
  base.py                # OnnxGraphOp 抽象クラス
  chain.py               # ChainOp（複数 op の直列合成）
  export_all.py          # モデル一括エクスポート + models_meta.json 生成
  onnx_cv_graph/         # 各オペレーション実装
models/                  # エクスポート先（自動生成）
tests/                   # pytest テスト
```

## セットアップ

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 使い方

### 1) モデルを生成

```bash
python src/export_all.py
```

生成物は `models/` に出力されます。

### 2) 推論の最小例

```python
import cv2
import numpy as np
import onnxruntime as ort

bgr = cv2.imread("assets/sample.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
img = rgb.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0  # (1,3,H,W)

session = ort.InferenceSession("models/grayscale.onnx")
out = session.run(None, {"input": img})[0]
```

### 3) 複数オペレーションを 1 モデルにまとめる

```python
from src.chain import ChainOp
from src.onnx_cv_graph import GrayscaleOp, BinarizeOp

chain = ChainOp([GrayscaleOp(), BinarizeOp()])
chain.export("models/grayscale_binarize.onnx")
```

## テスト

```bash
pytest tests/ -v
```

テスト内容の詳細は [TEST_DESIGN.md](TEST_DESIGN.md) を参照してください。

## テンソル規約

全オペレーション共通の規約です。

| 項目 | 規約 |
|------|------|
| レイアウト | NCHW |
| データ型 | float32 |
| チャネル順 | RGB |
| 値域 | [0,1] |

実装時の注意点:

- 規約から外れる形式の入出力はモデル外で変換する
- 出力が [0,1] を超える可能性がある処理は `Clip(0,1)` を入れる
- 2値化やグレースケールでも、出力は 3ch で統一する

## 2入力オペレーション

2枚の画像を受けるオペレーションは、入力名を次のようにそろえます。

| 入力名 | 役割 |
|--------|------|
| `input` | 主画像 |
| `input2` | 副画像 |

## パラメータメタデータ

外部から調整するパラメータ（例: threshold）がある場合は、`metadata_props` に最小値・最大値・既定値を入れます。

| キー | 値の形式 | 例 |
|------|-----------|----|
| `param:{name}` | `min,max,default` | `param:threshold = 0.0,1.0,0.5` |

## Web デモ

```bash
python src/export_all.py
python -m http.server 8080
# http://localhost:8080/sample_app.html
```

## 追加実装の手順

1. `src/onnx_cv_graph/` に `OnnxGraphOp` サブクラスを追加
2. `op_name`, `input_specs`, `output_specs`, `build_graph` を実装
3. 必要なら `param_meta` と `variants()` を実装
4. `src/onnx_cv_graph/__init__.py` に import を追加
5. `tests/test_{op_name}.py` を追加
6. `python src/export_all.py` と `pytest tests/ -v` で確認
