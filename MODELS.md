# ONNX オペレーションで実現可能な画像処理検討

ONNX opset 17 のオペレーターのみで実現可能な OpenCV 相当の画像処理を検討・列挙する。

凡例:
- 難易度
    - ★ 簡単 (1-2ノード)
    - ★★ 普通 (3-5ノード)
    - ★★★ やや複雑 (6-9ノード)
    - ★★★★ 複雑 (10-19ノード)
    - ★★★★★ かなり複雑 (20+ノード)
- ✅ = 実装済み
- 実装名 = `op_name` (= モデルファイル名 `{op_name}.onnx`)

---

## 1. ピクセル単位演算 (element-wise)

| 処理 | 実装名 | OpenCV 相当 | ONNX ノード構成 | 難易度 | パラメータ | 状態 |
|------|--------|-------------|----------------|--------|-----------|------|
| グレースケール | `grayscale` | `cvtColor(BGR2GRAY)` | Mul → ReduceSum → Expand | ★★ | — | ✅ |
| 閾値2値化 | `binarize` | `threshold(THRESH_BINARY)` | Mul → ReduceSum → Greater → Cast → Expand | ★★ | threshold | ✅ |
| 明るさ調整 | `brightness` | `add` | Add → Clip | ★ | brightness | ✅ |
| コントラスト調整 | `contrast` | `convertScaleAbs` | Sub → Mul → Add → Clip | ★★ | contrast, center | ✅ |
| ガンマ補正 | `gamma` | `LUT` (ガンマテーブル) | Pow → Clip | ★ | gamma | ✅ |
| ネガポジ反転 | `invert` | `bitwise_not` | Sub (1.0 - input) | ★ | — | ✅ |
| ソラリゼーション | `solarize` | — | GreaterOrEqual → Sub → Where | ★★ | threshold | ✅ |
| ポスタリゼーション | `posterize` | — | Mul → Floor → Div → Clip | ★★ | levels | ✅ |
| 彩度調整 | `saturation` | — | Mul → ReduceSum → Expand → Sub → Mul → Mul → Add → Clip | ★★★ | saturation | ✅ |
| 露出調整 | `exposure` | — | Mul → Clip | ★ | exposure | ✅ |
| レベル補正 | `levels` | Photoshop Levels | Clip(in) → Sub → Sub → Max(ε) → Div → Clip → Pow → Sub → Mul → Add → Clip | ★★★★ | in_black, in_white, gamma, out_black, out_white | ✅ |
| オートレベル補正 | `auto_levels` | Photoshop 自動レベル補正 | ReduceMin → ReduceMax → Sub → Max(ε) → Sub → Div → Clip | ★★★ | — | ✅ |
| クリッピング | | `min`/`max` | Clip | ★ | min, max |  |
| 絶対値 | | `absdiff` | Sub → Abs | ★ | — |  |
| 加算 | | `add` | Add → Clip | ★ | — |  |
| 減算 | | `subtract` | Sub → Clip | ★ | — |  |
| 乗算 | | `multiply` | Mul | ★ | scale |  |
| 除算 | | `divide` | Div | ★ | scale |  |
| 最大値 | | `max` | Max | ★ | — |  |
| 最小値 | | `min` | Min | ★ | — |  |
| 累乗 | | `pow` | Pow | ★ | power |  |
| 平方根 | | `sqrt` | Sqrt | ★ | — |  |
| 対数 | | `log` | Log | ★ | — |  |
| シグモイドトーン | | — | Sigmoid → 線形スケーリング | ★ | gain, cutoff |  |

## 2. 色空間変換

| 処理 | 実装名 | OpenCV 相当 | ONNX ノード構成 | 難易度 | パラメータ | 状態 |
|------|--------|-------------|----------------|--------|-----------|------|
| RGB→BGR | `rgb2bgr` | `cvtColor(RGB2BGR)` | Gather (チャネル軸で [2,1,0]) | ★ | — | ✅ |
| セピア調 | `sepia` | 色変換行列 | Conv(1×1) → Clip | ★ | — | ✅ |
| RGB→YCrCb | | `cvtColor(RGB2YCrCb)` | Conv(1×1) → Add | ★★ | — |  |
| YCrCb→RGB | | `cvtColor(YCrCb2RGB)` | Sub → Conv(1×1) | ★★ | — |  |
| RGB→YUV | | `cvtColor(RGB2YUV)` | Conv(1×1) | ★★ | — |  |
| HSV 抽出 (H/S/V) | `hsv_h` `hsv_s` `hsv_v` | `cvtColor(RGB2HSV)` | Gather → Max/Min → Sub → Div → Equal → Where → Clip → Expand | ★★★★★ | channel (3種: h/s/v) | ✅ |
| RGB→HLS | | `cvtColor(RGB2HLS)` | ReduceMax → ReduceMin → Sub → Div → Equal → Where | ★★★ | — |  |
| チャネル抽出 | `channel_r` `channel_g` `channel_b` | `split` 相当 | Gather(axis=1) → Expand | ★ | channel (3種: r/g/b) | ✅ |
| HSV 範囲抽出 | `hsv_range` | `cvtColor` + `inRange` | RGB→HSV → GreaterOrEqual/LessOrEqual → And → Cast → Expand | ★★★★★ | h/s/v_min, h/s/v_max | ✅ |
| カラーマップ適用 | `colormap_jet` `colormap_turbo` `colormap_inferno` `colormap_viridis` | `applyColorMap` | Mul → ReduceSum → Mul → Floor → Cast → Clip → Squeeze → Gather(LUT) → Transpose | ★★★ | colormap (4種) | ✅ |
| 色温度調整 | `color_temperature` | — | Add → Sub → Concat → Unsqueeze → Mul → Clip | ★★★ | temperature | ✅ |
| WB チャネルゲイン | `wb_gain` | — | Concat → Unsqueeze → Mul → Clip | ★★ | r_gain, g_gain, b_gain | ✅ |
| WB Gray World | `wb_gray_world` | `xphoto::GrayworldWB` | ReduceMean → ReduceMean → Add → Div → Mul → Clip | ★★★ | — | ✅ |
| WB White Patch | `wb_white_patch` | `xphoto::SimpleWB` | ReduceMax → Add → Div → Mul → Clip | ★★ | — | ✅ |
| 色抑制 (スタンプ除去) | `color_suppress` | — | RGB→HSV → 色相距離+彩度マスク → 白置換 | ★★★★★ | h_center, h_range, s_min, strength | ✅ |
| 色行列変換 (汎用) | | — | Conv(1×1, 3→3 任意行列) → Clip | ★★ | 3×3 matrix |  |

## 3. 畳み込みフィルタ (Conv with fixed kernel)

ONNX の Conv はカーネル shape がグラフ定義時に固定されるため、カーネルサイズの動的変更はできない。
よく使うサイズごとに個別の ONNX モデルを生成する方針とする。

- **エッジ検出系** (Sobel, Scharr, Prewitt, Laplacian): **3×3 固定** (実用上ほぼこれのみ)
- **ぼかし系** (平均, ガウシアン): **3×3 / 5×5 / 7×7** の3種
- **シャープ・エンボス**: **3×3 固定**
- ファイル名例: `gaussian_blur_3x3.onnx`, `gaussian_blur_5x5.onnx`, `gaussian_blur_7x7.onnx`

| 処理 | 実装名 | OpenCV 相当 | ONNX ノード構成 | 難易度 | カーネルサイズ | 状態 |
|------|--------|-------------|----------------|--------|--------------|------|
| 平均ぼかし | `blur_3x3` `blur_5x5` `blur_7x7` | `blur` / `boxFilter` | Pad → Conv(v) → Conv(h) | ★ | 3×3 / 5×5 / 7×7 | ✅ |
| ガウシアンぼかし | `gaussian_blur_3x3` `gaussian_blur_5x5` `gaussian_blur_7x7` | `GaussianBlur` | Pad → Conv(v) → Conv(h) | ★ | 3×3 / 5×5 / 7×7 | ✅ |
| シャープ化 | `sharpen` | `filter2D` (シャープカーネル) | Pad → Conv → Clip | ★★ | 3×3 | ✅ |
| エンボス | `emboss` | `filter2D` (エンボスカーネル) | Pad → Conv → Add → Clip | ★★ | 3×3 | ✅ |
| Sobel エッジ | `sobel` | `Sobel` | Mul → ReduceSum → Pad → Conv(xy) → Split → Abs → Abs → Add → Clip → Expand | ★★★★ | 3×3 | ✅ |
| Scharr エッジ | `scharr` | `Scharr` | Mul → ReduceSum → Pad → Conv(xy) → Split → Abs → Abs → Add → Div → Clip → Expand | ★★★★ | 3×3 | ✅ |
| Laplacian エッジ | `laplacian` | `Laplacian` | Mul → ReduceSum → Pad → Conv → Abs → Div → Clip → Expand | ★★★ | 3×3 | ✅ |
| Prewitt エッジ | `prewitt` | `filter2D` | Mul → ReduceSum → Pad → Conv(xy) → Split → Abs → Abs → Add → Div → Clip → Expand | ★★★★ | 3×3 | ✅ |
| アンシャープマスク | `unsharp_mask_3x3` `unsharp_mask_5x5` `unsharp_mask_7x7` | `GaussianBlur` + 加算 | Pad → Conv(v) → Conv(h) → Sub → Mul → Add → Clip | ★★★ | 3×3 / 5×5 / 7×7 | ✅ |
| エッジ強度 (magnitude) | `edge_magnitude` | `magnitude` | Mul → ReduceSum → Pad → Conv(xy) → Split → Mul → Mul → Add → Sqrt → Div → Clip → Expand | ★★★★ | 3×3 | ✅ |
| 背景ムラ補正 | `bg_normalize_3x3` `bg_normalize_5x5` `bg_normalize_7x7` | — | Pad → Conv(v) → Conv(h) → Sub → Add(0.5) → Clip | ★★ | 3×3 / 5×5 / 7×7 | ✅ |
| Difference of Gaussians | `dog_3x3_5x5` `dog_3x3_7x7` `dog_5x5_7x7` | — | Mul → ReduceSum → Pad → Conv(σ1_v) → Conv(σ1_h) → Pad → Conv(σ2_v) → Conv(σ2_h) → Sub → Abs → Div → Clip → Expand | ★★★★ | 3×3/5×5, 3×3/7×7, 5×5/7×7 | ✅ |
| Laplacian of Gaussian | `log_5x5` `log_7x7` | — | Mul → ReduceSum → Pad → Conv(LoG) → Abs → Div → Clip → Expand | ★★★ | 5×5 / 7×7 | ✅ |
| Kuwahara フィルタ | `kuwahara_5x5` `kuwahara_7x7` `kuwahara_9x9` | — | Mul → ReduceSum (輝度) → Mul (自乗) → AveragePool×12 (4象限×gray/gray²/RGB) → Sub (分散) → Concat → ReduceMin → Equal×4 → Where×3 → Clip | ★★★★★ | 5×5 / 7×7 / 9×9 | ✅ |
| XDoG (スケッチ) | `xdog_3x3_5x5` `xdog_5x5_9x9` `xdog_7x7_13x13` | — | Mul → ReduceSum (輝度) → Pad → Conv(σ1) → Pad → Conv(σ2) → Mul+Sub (DoG) → ReduceMax+Max+Div (正規化) → Sub+Mul+Tanh+Add (XDoG) → Clip → Expand | ★★★★ | 3×3/5×5, 5×5/9×9, 7×7/13×13 | ✅ |

## 4. モルフォロジー演算

MaxPool のカーネルサイズもグラフ定義時に固定。**3×3 / 5×5** の2種を生成する。

| 処理 | 実装名 | OpenCV 相当 | ONNX ノード構成 | 難易度 | カーネルサイズ | 状態 |
|------|--------|-------------|----------------|--------|--------------|------|
| 膨張 | `dilate_3x3` `dilate_5x5` | `dilate` | MaxPool | ★ | 3×3 / 5×5 | ✅ |
| 収縮 | `erode_3x3` `erode_5x5` | `erode` | Neg → MaxPool → Neg | ★★ | 3×3 / 5×5 | ✅ |
| オープニング | `opening_3x3` `opening_5x5` | `morphologyEx(OPEN)` | Neg → MaxPool → Neg → MaxPool | ★★ | 3×3 / 5×5 | ✅ |
| クロージング | `closing_3x3` `closing_5x5` | `morphologyEx(CLOSE)` | MaxPool → Neg → MaxPool → Neg | ★★ | 3×3 / 5×5 | ✅ |
| 勾配 | `gradient_3x3` `gradient_5x5` | `morphologyEx(GRADIENT)` | MaxPool → Neg → MaxPool → Neg → Sub | ★★ | 3×3 / 5×5 | ✅ |
| トップハット | `tophat_3x3` `tophat_5x5` | `morphologyEx(TOPHAT)` | Neg → MaxPool → Neg → MaxPool → Sub | ★★ | 3×3 / 5×5 | ✅ |
| ブラックハット | `blackhat_3x3` `blackhat_5x5` | `morphologyEx(BLACKHAT)` | MaxPool → Neg → MaxPool → Neg → Sub | ★★ | 3×3 / 5×5 | ✅ |
| ヒットオアミス | `hitmiss_3x3` | `morphologyEx(HITMISS)` | Mul → ReduceSum → GreaterOrEqual → Cast → Conv(fg) → Equal → Sub → Conv(bg) → Equal → And → Cast → Expand | ★★★★ | 3×3 | ✅ |

## 5. 幾何変換

| 処理 | 実装名 | OpenCV 相当 | ONNX ノード構成 | 難易度 | パラメータ | 状態 |
|------|--------|-------------|----------------|--------|-----------|------|
| リサイズ (倍率) | `resize` | `resize` | Concat → Resize | ★ | scale | ✅ |
| リサイズ (任意サイズ) | `resize_to` | `resize` | Shape → Slice → Cast → Concat → Resize | ★★★ | target_h, target_w | ✅ |
| 水平反転 | `hflip` | `flip(1)` | Slice (step=-1) | ★ | — | ✅ |
| 垂直反転 | `vflip` | `flip(0)` | Slice (step=-1) | ★ | — | ✅ |
| 上下左右反転 | `hvflip` | `flip(-1)` | Slice (H逆順) → Slice (W逆順) | ★ | — | ✅ |
| 90°回転 | `rotate_90` | `rotate(ROTATE_90)` | Transpose → Slice (step=-1) | ★ | — | ✅ |
| 180°回転 | `rotate_180` | `rotate(ROTATE_180)` | Slice (H逆順) → Slice (W逆順) | ★ | — | ✅ |
| 270°回転 | `rotate_270` | `rotate(ROTATE_270)` | Transpose → Slice (step=-1) | ★ | — | ✅ |
| 任意角度回転 | `rotate_arbitrary` | `getRotationMatrix2D` + `warpAffine` | meshgrid → 回転行列適用 → GridSample → Clip | ★★★★ | angle | ✅ |
| 任意領域クロップ | `crop` | ROI 切り出し | Shape → Gather → Cast → Mul → Floor → Cast → Add → Min → Reshape → Concat → Slice | ★★★★★ | crop_top, crop_left, crop_h, crop_w | ✅ |
| 中央クロップ | `center_crop` | ROI 切り出し | Shape → Gather → Cast → Mul → Floor → Sub → Div → Cast → Add → Reshape → Concat → Slice | ★★★★★ | crop_ratio | ✅ |
| アフィン変換 | `affine` | `warpAffine` | meshgrid → 行列適用 → GridSample → Clip | ★★★ | a, b, tx, c, d, ty (*1) | ✅ |
| 射影変換 | `perspective` | `warpPerspective` | meshgrid → ホモグラフィ適用 → GridSample → Clip | ★★★★ | p00..p21 (*2) | ✅ |
| パディング (鏡像反転) | `padding_reflect` | `copyMakeBorder(REFLECT)` | Shape → Gather → Cast → Mul → Floor → Cast → Concat → Pad(reflect) | ★★★★ | pad_ratio | ✅ |
| パディング (任意色) | `padding_color` | `copyMakeBorder(CONSTANT)` | Shape → Cast → Mul → Floor → Concat → Pad(constant=0) → ConstantOfShape → Pad → Reshape → Sub → Mul → Add | ★★★★★ | pad_ratio, pad_r/g/b | ✅ |
| ピラミッドダウン | `pyr_down` | `pyrDown` | Pad → Conv (5×5 Gaussian) → Resize (½) | ★★ | — | ✅ |
| ピラミッドアップ | `pyr_up` | `pyrUp` | Resize (×2) → Pad → Conv (5×5 Gaussian) → Clip | ★★ | — | ✅ |
| タイリング (繰り返し) | | — | Tile | ★ | repeat_h, repeat_w |  |

> **(*1) アフィン変換のパラメータ**: 逆マッピング行列 `[[a, b, tx], [c, d, ty]]` を指定する。
> 座標系は [-1, 1] に正規化されており、出力の各ピクセル (x, y) に対して入力の参照先を `x_src = a*x + b*y + tx`, `y_src = c*x + d*y + ty` で計算する。
>
> | パラメータ | 意味 | 単位恒等変換 |
> |-----------|------|------------|
> | a | X方向スケール (水平) | 1.0 |
> | b | Y→X のせん断 (水平シアー) | 0.0 |
> | tx | X方向平行移動 | 0.0 |
> | c | X→Y のせん断 (垂直シアー) | 0.0 |
> | d | Y方向スケール (垂直) | 1.0 |
> | ty | Y方向平行移動 | 0.0 |
>
> 例: `a=0.5, d=0.5` → 中央に 50% 縮小、`b=0.3` → 水平せん断、`tx=0.2` → 右に 20% シフト
>
> **(*2) 射影変換のパラメータ**: ホモグラフィ行列の 8 要素 `[[p00, p01, p02], [p10, p11, p12], [p20, p21, 1]]` を指定する。
> 出力の各ピクセル (x, y) に対して、入力の参照先を以下で計算する:
> ```
> w     = p20*x + p21*y + 1
> x_src = (p00*x + p01*y + p02) / w
> y_src = (p10*x + p11*y + p12) / w
> ```
>
> | パラメータ | 意味 | 単位恒等変換 |
> |-----------|------|------------|
> | p00 | X方向スケール | 1.0 |
> | p01 | Y→X のせん断 | 0.0 |
> | p02 | X方向平行移動 | 0.0 |
> | p10 | X→Y のせん断 | 0.0 |
> | p11 | Y方向スケール | 1.0 |
> | p12 | Y方向平行移動 | 0.0 |
> | p20 | 水平方向の射影成分 (台形化) | 0.0 |
> | p21 | 垂直方向の射影成分 (台形化) | 0.0 |
>
> p00〜p12 はアフィン変換と同じ役割。p20, p21 が射影特有の成分で、遠近感（台形変形）を制御する。

## 6. 正規化・統計

| 処理 | 実装名 | OpenCV 相当 | ONNX ノード構成 | 難易度 | パラメータ | 状態 |
|------|--------|-------------|----------------|--------|-----------|------|
| Min-Max 正規化 | `minmax_norm` | `normalize(NORM_MINMAX)` | ReduceMin → ReduceMax → Sub → Max(ε) → Sub → Div | ★★★ | — | ✅ |
| L2 正規化 (全体) | `l2_norm` | `normalize(NORM_L2)` | Mul → ReduceSum → Sqrt → Add(ε) → Div → Clip | ★★★ | — | ✅ |
| L1 正規化 (全体) | `l1_norm` | `normalize(NORM_L1)` | Abs → ReduceSum → Add(ε) → Div → Clip | ★★ | — | ✅ |
| L2 正規化 (チャネル) | `l2_norm_ch` | — | Mul → ReduceSum(C) → Sqrt → Add(ε) → Div → Clip | ★★★ | — | ✅ |
| L1 正規化 (チャネル) | `l1_norm_ch` | — | Abs → ReduceSum(C) → Add(ε) → Div → Clip | ★★ | — | ✅ |
| 局所コントラスト正規化 | `lcn_15x15` `lcn_31x31` | — | AveragePool(μ) → AveragePool(gray²) → Sqrt(σ) → (x−μ)/(σ+ε) → Sigmoid | ★★★ | 15×15 / 31×31 | ✅ |
| 平均・標準偏差 | | `meanStdDev` | ReduceMean → Sub → Mul → ReduceMean → Sqrt | ★★ | — |  |
| 平均正規化 | | — | ReduceMean → Sub | ★ | — |  |
| 標準化 (z-score) | | — | ReduceMean → Sub → Sqrt(Var) → Div | ★★ | — |  |

## 7. ブレンド・合成

2入力オペレーション。入力は `input` (主画像) と `input2` (副画像) の命名規則に従う。

| 処理 | 実装名 | OpenCV 相当 | ONNX ノード構成 | 難易度 | パラメータ | 状態 |
|------|--------|-------------|----------------|--------|-----------|------|
| 加重加算 | `weighted_add` | `addWeighted` | Mul → Mul → Add → Add → Clip | ★★ | alpha, beta, gamma | ✅ |
| アルファブレンド | `alpha_blend` | — | Mul → Sub → Mul → Add | ★★ | alpha | ✅ |
| マスク合成 | `mask_composite` | `copyTo` (mask) | Mul → Sub → Mul → Add | ★★ | — | ✅ |
| オーバーレイ | `overlay` | — | Mul → Mul → Sub → Sub → Mul → Mul → Sub → Less → Where → Clip | ★★★★ | — | ✅ |

## 8. 閾値処理バリエーション

| 処理 | 実装名 | OpenCV 相当 | ONNX ノード構成 | 難易度 | パラメータ | 状態 |
|------|--------|-------------|----------------|--------|-----------|------|
| 逆2値化 | `inv_binarize` | `threshold(THRESH_BINARY_INV)` | Mul → ReduceSum → Greater → Cast → Sub → Expand | ★★★ | threshold | ✅ |
| 切り詰め | `thresh_trunc` | `threshold(THRESH_TRUNC)` | Mul → ReduceSum → Min → Expand | ★★ | threshold | ✅ |
| ゼロ化 | `thresh_tozero` | `threshold(THRESH_TOZERO)` | Mul → ReduceSum → Greater → Cast → Mul → Expand | ★★★ | threshold | ✅ |
| 逆ゼロ化 | `thresh_tozero_inv` | `threshold(THRESH_TOZERO_INV)` | Mul → ReduceSum → Greater → Not → Cast → Mul → Expand | ★★★ | threshold | ✅ |
| 適応的閾値 (平均) | `adaptive_thresh_mean_3x3` `…5x5` `…7x7` | `adaptiveThreshold(MEAN_C)` | Mul → ReduceSum → AveragePool → Sub → Greater → Cast → Expand | ★★★ | C (block_size 固定: 3/5/7) | ✅ |
| 適応的閾値 (ガウス) | `adaptive_thresh_gaussian_3x3` `…5x5` `…7x7` | `adaptiveThreshold(GAUSSIAN_C)` | Mul → ReduceSum → Pad → Conv(v) → Conv(h) → Sub → Greater → Cast → Expand | ★★★ | C (block_size 固定: 3/5/7) | ✅ |
| Sauvola 局所適応2値化 | `sauvola_15x15` `sauvola_31x31` `sauvola_63x63` | — | Mul → ReduceSum (輝度) → AveragePool(μ) → AveragePool(gray²) → Sqrt(σ) → T=μ×(1+k×(σ/R−1)) → Greater → Expand | ★★★★ | k (15×15 / 31×31 / 63×63) | ✅ |
| 範囲内抽出 | `inrange` | `inRange` | Mul → ReduceSum → GreaterOrEqual → LessOrEqual → And → Cast → Expand | ★★★ | lower, upper | ✅ |

> **注**: 適応的閾値の block_size は AveragePool / Conv のカーネルサイズとなるためグラフ定義時に固定。
> block_size ごとに別モデルを生成する（3/5/7 の3種）。C は推論時パラメータとして渡す。

## 9. 特徴量・コーナー検出

| 処理 | 実装名 | OpenCV 相当 | ONNX ノード構成 | 難易度 | パラメータ | 状態 |
|------|--------|-------------|----------------|--------|-----------|------|
| Harris コーナー | `harris_corner_3x3` `harris_corner_5x5` | `cornerHarris` | Mul → ReduceSum → Pad → Conv(Sobel xy) → Split → Mul(Ix²,Iy²,IxIy) → AveragePool(窓) → det/trace → MinMax正規化 → Clip → Expand | ★★★★★ | block_size, k | ✅ |
| Shi-Tomasi スコア | `shi_tomasi_3x3` `shi_tomasi_5x5` | `cornerMinEigenVal` | Mul → ReduceSum → Pad → Conv(Sobel xy) → Split → Mul → AveragePool(窓) → 最小固有値 → Sqrt → MinMax正規化 → Clip → Expand | ★★★★★ | block_size | ✅ |
| 直線抽出 (形態学的) | `line_extract_h_15` `line_extract_v_15` | 形態学的演算 | Mul → ReduceSum → Mul → MaxPool → Mul → MaxPool → Clip → Expand | ★★★ | line_length | ✅ |

## 10. ML 前処理 (機械学習パイプライン向け)

推論前の正規化・チャネル変換など、ML モデルへの入力整形に使う処理群。
ChainOp で合成すれば、画像読み込み→前処理→ML推論 の前段を1つの ONNX モデルにまとめられる。

| 処理 | 実装名 | 用途 | ONNX ノード構成 | 難易度 | 状態 |
|------|--------|------|----------------|--------|---------|
| ImageNet 正規化 | `imagenet_norm` | torchvision `Normalize(mean, std)` | Sub(mean) → Div(std) | ★ | ✅ |
| [0,1]→[0,255] スケーリング | `scale_to_255` | uint8 出力相当への変換 | Mul(255) | ★ | ✅ |
| [0,255]→[0,1] スケーリング | `scale_from_255` | uint8 入力の正規化 | Div(255) | ★ | ✅ |
| [-1,1] 正規化 | `normalize_neg1_pos1` | GAN / CLIP 等の入力 | Mul(2) → Sub(1) | ★ | ✅ |
| チャネルごと平均減算 | `channel_mean_sub` | Caffe 系モデルの前処理 | Sub(mean_per_channel) | ★ | ✅ |
| ピクセル平均減算 | `pixel_mean_sub` | detectron2/Caffe 系 `*255 - [123.675, 116.28, 103.53]` | Mul(255) → Sub(pixel_mean) | ★ | ✅ |
| リサイズ + パディング (letterbox) | `letterbox` | YOLO 等の入力整形 | Shape → Gather → Cast → Div → Min → Concat → Reshape → Gather → Cast → Sub → Div → Floor → Cast → Reshape → Concat → Pad | ★★★★★ | ✅ |
| センタークロップ | `center_crop` | ViT / ResNet 等の推論入力 | Shape → Gather → Cast → Mul → Floor → Sub → Div → Cast → Add → Reshape → Concat → Slice | ★★★★★ | ✅ |
| ランダムクロップ相当 (固定オフセット) | | テスト時の再現性あるクロップ | Slice | ★ |  |
| HWC→CHW 変換 | `hwc_to_chw` | NumPy/PIL 画像→テンソル変換 | Transpose (0,3,1,2) | ★ | ✅ |
| CHW→HWC 変換 | `chw_to_hwc` | テンソル→画像表示用 | Transpose (0,2,3,1) | ★ | ✅ |
| float32→uint8 量子化 | `float_to_uint8` | 後処理での画像出力 | Mul(255) → Clip(0,255) → Round → Cast(uint8) | ★★ | ✅ |
| uint8→float32 変換 | `uint8_to_float` | uint8 画像の正規化 | Cast(float32) → Div(255) | ★ | ✅ |
| バッチ次元追加 (NCHW) | `batch_unsqueeze_nchw` | (3,H,W) → (1,3,H,W) | Unsqueeze (axis=0) | ★ | ✅ |
| バッチ次元追加 (NHWC) | `batch_unsqueeze_nhwc` | (H,W,3) → (1,H,W,3) | Unsqueeze (axis=0) | ★ | ✅ |
| バッチ次元除去 (NCHW) | `batch_squeeze_nchw` | (1,3,H,W) → (3,H,W) | Squeeze (axis=0) | ★ | ✅ |
| バッチ次元除去 (NHWC) | `batch_squeeze_nhwc` | (1,H,W,3) → (H,W,3) | Squeeze (axis=0) | ★ | ✅ |

---

## 実現困難な処理 (参考)

以下は ONNX オペレーションだけでは実現が難しい、または非実用的な処理:

| 処理 | OpenCV 相当 | 理由 |
|------|-------------|------|
| Canny エッジ検出 | `Canny` | 非最大抑制 + ヒステリシス閾値のループが必要 |
| ヒストグラム均一化 | `equalizeHist` | 累積ヒストグラム計算にソート/スキャンが必要 |
| CLAHE | `createCLAHE` | タイルごとのヒストグラム均一化 + 補間が必要 |
| 大津の閾値 | `threshold(THRESH_OTSU)` | ヒストグラム分析の反復処理が必要 |
| Hough 変換 (直線) | `HoughLines` / `HoughLinesP` | 投票空間の反復処理が必要 |
| Hough 変換 (円) | `HoughCircles` | 投票空間の反復処理が必要 |
| テンプレートマッチング | `matchTemplate` | スライディングウィンドウのループが必要 |
| リマッピング | `remap` | 任意座標マッピングが必要 |
| 輪郭検出 | `findContours` | 連結成分探索のループが必要 |
| 輪郭近似 | `approxPolyDP` | Douglas-Peucker の再帰処理が必要 |
| 凸包 | `convexHull` | ソート + スタック操作が必要 |
| モーメント | `moments` | 輪郭依存の集計処理 |
| 距離変換 | `distanceTransform` | 反復的な距離伝搬が必要 |
| Watershed | `watershed` | マーカーベースの反復セグメンテーション |
| GrabCut | `grabCut` | GMM + グラフカットの反復最適化 |
| 塗りつぶし | `floodFill` | 再帰的/キュー的な領域探索が必要 |
| 修復 (Inpaint) | `inpaint` | PDE ベースの反復処理が必要 |
| 非局所平均フィルタ | `fastNlMeansDenoising` | ブロックマッチングの全探索が必要 |
| バイラテラルフィルタ | `bilateralFilter` | ピクセルごとの空間+輝度重み計算が必要 |
| メディアンフィルタ | `medianBlur` | ウィンドウ内のソート (中央値) が必要 |
| 特徴点検出 (SIFT等) | `SIFT` / `ORB` / `AKAZE` | スケールスペース + 極値検出 + 記述子計算 |
| オプティカルフロー | `calcOpticalFlowFarneback` | 反復的な最適化が必要 |
| 背景差分 | `createBackgroundSubtractorMOG2` | 統計モデルの逐次更新が必要 |
| カラー転写 | `seamlessClone` | ポアソン方程式の反復求解が必要 |

