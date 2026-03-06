import cv2
import numpy as np
import onnxruntime as ort

from src.chain import ChainOp
from src.onnx_cv_graph import (
    BatchSqueezeNhwcOp,
    BatchUnsqueezeNhwcOp,
    ChwToHwcOp,
    GrayscaleOp,
    HwcToChwOp,
    Rgb2BgrOp,
    ScaleFrom255Op,
    ScaleTo255Op,
)

# --- パイプライン定義 ---
pipeline = ChainOp([
    BatchUnsqueezeNhwcOp(),  # バッチ次元追加
    HwcToChwOp(),            # HWC → CHW
    ScaleFrom255Op(),        # [0,255] → [0,1]
    Rgb2BgrOp(),             # BGR → RGB
    GrayscaleOp(),           # グレースケール化
    Rgb2BgrOp(),             # RGB → BGR
    ScaleTo255Op(),          # [0,1] → [0,255]
    ChwToHwcOp(),            # CHW → HWC
    BatchSqueezeNhwcOp(),    # バッチ次元除去
])

# --- ONNX モデルのエクスポート ---
model_path = "models/example/pipeline.onnx"
pipeline.export(model_path)

# --- セッションのロード ---
sess = ort.InferenceSession(model_path)

# --- 画像の読み込み ---
bgr = cv2.imread("assets/sample.jpg")
img = bgr.astype(np.float32)

# --- 推論 ---
out = sess.run(None, {"input": img})[0]

# --- uint8 に変換 ---
result = out.astype(np.uint8)

# --- 結果表示 ---
cv2.imshow("input",  bgr)
cv2.imshow("output", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
