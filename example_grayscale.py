import cv2
import numpy as np
import onnxruntime as ort

# --- セッションのロード ---
# 前処理
sess_unsqueeze  = ort.InferenceSession("models/10_ml_preprocess/batch_unsqueeze_nhwc.onnx")
sess_hwc2chw    = ort.InferenceSession("models/10_ml_preprocess/hwc_to_chw.onnx")
sess_from255    = ort.InferenceSession("models/10_ml_preprocess/scale_from_255.onnx")
sess_bgr2rgb    = ort.InferenceSession("models/02_color_space/rgb2bgr.onnx")
# メイン処理
sess_grayscale  = ort.InferenceSession("models/01_elementwise/grayscale.onnx")
# 後処理
sess_rgb2bgr    = ort.InferenceSession("models/02_color_space/rgb2bgr.onnx")  # 同じモデルで逆変換
sess_to255      = ort.InferenceSession("models/10_ml_preprocess/scale_to_255.onnx")
sess_chw2hwc    = ort.InferenceSession("models/10_ml_preprocess/chw_to_hwc.onnx")
sess_squeeze    = ort.InferenceSession("models/10_ml_preprocess/batch_squeeze_nhwc.onnx")

# --- 画像の読み込み ---
bgr = cv2.imread("assets/sample.jpg")   # (H,W,3) uint8、OpenCV は BGR で読み込む

# --- 前処理 ---
img = bgr.astype(np.float32)
img = sess_unsqueeze.run(None, {"input": img})[0]
img = sess_hwc2chw.run(None, {"input": img})[0]
img = sess_from255.run(None, {"input": img})[0]
img = sess_bgr2rgb.run(None, {"input": img})[0]

# --- メイン処理 ---
out = sess_grayscale.run(None, {"input": img})[0]

# --- 後処理 ---
out = sess_rgb2bgr.run(None, {"input": out})[0]
out = sess_to255.run(None, {"input": out})[0]
out = sess_chw2hwc.run(None, {"input": out})[0]
out = sess_squeeze.run(None, {"input": out})[0]
result = out.astype(np.uint8)

# --- 結果表示 ---
cv2.imshow("input",  bgr)
cv2.imshow("output", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
