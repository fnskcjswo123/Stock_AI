import sys
import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path
import time
import os
import math

labels_folder = Path("labels")
models_folder = Path("models")

# 폴더 존재 여부 확인
if not labels_folder.exists():
    sys.exit(f"Labels folder not found: {labels_folder}")
if not models_folder.exists():
    sys.exit(f"Models folder not found: {models_folder}")

features_list = []

for file in labels_folder.iterdir():
    if file.suffix == ".txt":
        with open(file, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                features = parts[1:]  # 첫번째 값은 라벨, 나머지가 feature
                features_list.append(features)

features_array = np.array(features_list, dtype=np.float32)  # shape (N, feature_size)
batch_size, feature_size = features_array.shape
print(f"Loaded features: {features_array.shape}")

onnx_files = [f for f in models_folder.iterdir() if f.suffix == ".onnx"]
if not onnx_files:
    sys.exit("No ONNX models found in the models folder.")

onnx_sessions = [ort.InferenceSession(f, providers=['CPUExecutionProvider']) for f in onnx_files]
print(f"Loaded {len(onnx_sessions)} ONNX models.")

def pulsar_avg(values, cut_ratio=0.15):
    sorted_values = np.sort(values)
    n = len(sorted_values)
    k = int(n * cut_ratio)
    if n - 2 * k <= 0:
        return np.mean(sorted_values)
    return np.mean(sorted_values[k:n - k])

prediction_table = []
max_table = 100
print("Press ESC to exit.")

while True:
    if cv2.waitKey(100) & 0xFF == 27:  # ESC 키
        break

    model_preds = []
    for session in onnx_sessions:
        input_name = session.get_inputs()[0].name
        pred = session.run(None, {input_name: features_array})[0]  # shape (N,1)
        model_preds.append(pred.flatten())

    # 모델별 평균
    model_preds = np.array(model_preds)
    avg_pred = model_preds.mean(axis=0)  # shape (N,)
    
    # 절사평균
    smooth_pred = pulsar_avg(avg_pred)
    prediction_table.append(smooth_pred)
    if len(prediction_table) > max_table:
        prediction_table.pop(0)

    img = np.zeros((600, 800, 3), dtype=np.uint8)
    for i in range(1, len(prediction_table)):
        x1, y1 = int((i-1)*(800/max_table)), int(600 - prediction_table[i-1]*600)
        x2, y2 = int(i*(800/max_table)), int(600 - prediction_table[i]*600)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, f"Average: {smooth_pred:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Stock Prediction", img)

cv2.destroyAllWindows()
