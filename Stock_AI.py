import sys
import math
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image, ImageGrab
import os
from pathlib import Path
import time

# 생성할 폴더 경로
model_folder_path = Path("models")
models_folder = r".\models"
labels_folder = r".\labels"

# models 폴더 없으면 생성하고, 있으면 계속하기
if not model_folder_path.exists():
    sys.stdout.write("Folder does not exist.\n")
    sys.stdout.write(f"Creating directory: {model_folder_path}\n")
    model_folder_path.mkdir(parents=True)
else:
    sys.stdout.write("Folder already exists.\n")
sys.stdout.write(f"Folder path: {model_folder_path}\n")

# models 폴더 안에 있는 모든 ONNX 파일 리스트화
onnx_files = [f for f in model_folder_path.iterdir() if f.is_file() and f.suffix == '.onnx']

# onnx 파일 없으면 종료, 있으면 사용자가 파일 설정하기
if not onnx_files:
    sys.stdout.write("No ONNX models found in the 'models' folder.\n")
    sys.stdout.write("Please add at least one ONNX model file to proceed!\n")
    sys.stdout.write("The program will exit in 5 seconds...\n")
    time.sleep(5)
    sys.exit()

# models 폴더 안에 있는 모든 onnx 파일 로드
onnx_sessions = [ort.InferenceSession(f, providers=['CPUExecutionProvider']) for f in onnx_files]
sys.stdout.write(f"Loaded {len(onnx_sessions)} ONNX models.\n")

# 이미지 전처리
def image_process(img: Image.Image):
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, 0)
    return img

# 절사평균 구하기 (상하 15%)
def pulsar_avg(value,  pulsar_avg_value = 0.15):
    sorted_value = np.sort(value)
    n = len(value)
    k = int(n * pulsar_avg_value)
    if n - 2*k <= 0:
        return np.mean(sorted_value)
    pulsar = sorted_value[k:n-k]
    return np.mean(pulsar)


# 캡쳐 영역 설정
capture_section = (310, 540, 1610, 1080)
prediction_table = []
max_table = 100

# 메인 루프
sys.stdout.write("Press ESC to exit.\n")
while True:

    # ESC 키로 종료하기
    if cv2.waitKey(1) & 0xFF == 27: break

    # 화면 캡쳐
    screen = ImageGrab.grab(capture_section)
    input_tensor = image_process(screen)

    # 모든 onnx 모델 추론
    metatable = []
    for session in onnx_sessions:
        input_name = session.get_inputs()[0].name
        pred = session.run(None, {input_name: input_tensor.astype(np.float32)})[0]
        metatable.append(float(pred[0][0]))

    # 절사평균 적용
    prediction_value = pulsar_avg(metatable, 0.15)
    prediction_table.append(prediction_value)
    if len(prediction_table) > max_table:
        prediction_table.pop(0)

    # 화면 변환
    img_cv = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

    # 예측값 선 그래프 그리기
    for i in range(1, len(prediction_table)):
        x1, y1 = int((i-1)*(800/max_table)), int(600 - prediction_table[i-1]*600)
        x2, y2 = int(i*(800/max_table)), int(600 - prediction_table[i]*600)
        cv2.line(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # 절사평균 값 텍스트 표시
    cv2.putText(img_cv, f"Average: {prediction_value:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화면 출력
    cv2.imshow("Stock Prediction", img_cv)

cv2.destroyAllWindows()