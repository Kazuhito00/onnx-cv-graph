[[Japanese](README.md)/[English](README_EN.md)]

# onnx-cv-graph

ONNXグラフによる画像処理実装を検証する実験用リポジトリです。<br>
各処理はONNXエクスポートしたものを同梱しています。

<img width="1238" height="787" alt="image" src="https://github.com/user-attachments/assets/2d5c6039-be16-406a-8c7b-05e398d128ae" />

# Web Demo
エクスポートされたONNXを用いた推論デモは以下のページから確認できます。<br>
※Pythonの推論デモは後述
* https://kazuhito00.github.io/onnx-cv-graph/example_app.html

# Features
以下の特徴があります。
- 画像処理をONNXオペレータのみで実現
- 各処理はONNXファイルにエクスポートしてリポジトリに同梱
- 複数処理を直列に結合して一つのONNXファイルとして出力可能

# Purpose of This Repository
以下の検証を目的としています。
- ONNXオペレータのみで実現可能な画像処理の検証
- ONNX化によるクロスプラットフォーム・クロス言語での再現性確保
- ONNX RuntimeによるGPU/WebGL/TensorRT/DirectML等へのオフロード活用
- 複数画像処理を1つのONNXモデルに統合し、単一モデルとして配布

# Requirement
```
Python 3.10 or later

numpy          2.4.2     or later
onnx           1.20.1    or later
onnxruntime    1.24.2    or later
opencv-python  4.13.0.92 or later
pyvis          0.3.2     or later
pytest         9.0.2     or later
scipy          1.17.1    or later
```

# Installation

```bash
# リポジトリクローン
git clone https://github.com/Kazuhito00/onnx-cv-graph
cd onnx-cv-graph

# Python パッケージインストール
pip install -r requirements.txt
```

# ONNX Export

### ONNXエクスポート・テスト
すべてのONNXファイルをエクスポートする場合は以下のスクリプトを実行してください。<br>
テスト済みのONNXファイルはリポジトリに同梱しているため、推論だけ試す方は以下手順は不要です。
```bash
python src/export_all.py
python -m pytest tests/ -v
```

テスト方針は[TEST_DESIGN.md](TEST_DESIGN.md)を参照してください。

# Model List
実装済み・実装候補の画像処理は[MODELS.md](MODELS.md)に一覧化しています。

# Usage

### 実行例(Python)
グレースケール処理
```bash
python example_grayscale.py
```

ChainOpで単一ONNX化したグレースケール処理
```bash
python example_grayscale_chainop.py
```

### ONNX Runtime Webを用いたWebアプリデモ
```bash
python -m http.server 8080
# http://localhost:8080/example_app.html
```

# Project Structure

```text
README.md                # README（日本語）
README_EN.md             # README（英語）
MODELS.md                # 実装済み・実装候補 画像処理一覧
TEST_DESIGN.md           # テスト方針
requirements.txt         # Python依存パッケージ
example_app.html         # ONNX Runtime Webデモ
src/
  base.py                # OnnxGraphOp抽象クラス
  chain.py               # ChainOp（複数opの直列合成用クラス）
  export_all.py          # モデル一括エクスポート + models_meta.json生成
  onnx_cv_graph/         # 各グラフ実装
models/                  # ONNXファイル格納先
tests/                   # pytest用テストケース
assets/                  # サンプル画像・グラフ可視化HTML
```

# Author
高橋かずひと(https://x.com/KzhtTkhs)

# License
onnx-cv-graph is under [Apache-2.0 license](LICENSE).<br>

# License(Image)
サンプルの画像は[ぱくたそ](https://www.pakutaso.com/)様の「[猫背が治った！](https://www.pakutaso.com/20260129013post-56289.html)」を使用しています。
