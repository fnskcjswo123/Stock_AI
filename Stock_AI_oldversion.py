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

# models 폴더 없으면 생성하고, 있으면 계속하기
if not model_folder_path.exists():
    sys.stdout.write("Folder does not exist.\n")
    sys.stdout.write(f"Creating directory: {model_folder_path}\n")
    model_folder_path.mkdir(parents=True)
else:
    sys.stdout.write("Folder already exists.\n")
sys.stdout.write(f"Folder path: {model_folder_path}\n")

# models 폴더 안에 있는 onnx 파일 리스트화
model_folder = [i for i in model_folder_path.iterdir() if i.is_file() and i.suffix == '.onnx']

# onnx 파일 없으면 종료, 있으면 사용자가 파일 설정하기
if not model_folder:
    sys.stdout.write("No ONNX models found in the 'models' folder.\n")
    sys.stdout.write("Please add at least one ONNX model file to proceed!\n")
    sys.stdout.write("The program will exit in 5 seconds...\n")
    time.sleep(5)
    sys.exit()

# onnx 모델 있으면 사용가능한 모델 출력하기
sys.stdout.write("Availavle ONNX models:\n")
count = 1
for idk in model_folder:
    sys.stdout.write(f"{count}: {idk.name}\n")
    count += 1
sys.stdout.write("Select a model: ")

#사용자로부터 사용할 모델 입력받기
while True:
    userinput = sys.stdin.readline().strip()
    if not userinput.isdigit():
        sys.stdout.write("Please enter a number!\n")
        continue

    userinput = int(userinput)
    if userinput not in range(1, count):
        sys.stdout.write("Invalid number. Try again:\n")
        continue

    break

# onnx 모델 로드
onnx_model_path = model_folder[userinput - 1]
sys.stdout.write(f"Selected model: {onnx_model_path.name}\n")
onnx_session = ort.InferenceSession(onnx_model_path, providers=['GPUExecutionProvider', 'CPUExecutionProvider'])

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

    # ONNX 추론
    pred = onnx_session.run(None, {"input": input_tensor.astype(np.float32)})[0]
    prediction_value = float(pred[0][0])
    prediction_table.append(prediction_value)
    if len(prediction_table) > max_table: prediction_table.pop(0)

    # 절사평균 적용
    smooth_average = pulsar_avg(prediction_table, 0.15)

    # 화면 변환
    img_cv = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

    # 예측값 선 그래프 그리기
    for i in range(1, len(prediction_table)):
        x1, y1 = int((i-1)*(800/max_table)), int(600 - prediction_table[i-1]*600)
        x2, y2 = int(i*(800/max_table)), int(600 - prediction_table[i]*600)
        cv2.line(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 절사평균 값 텍스트 표시
    cv2.putText(img_cv, f"Average: {smooth_average:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 화면 출력
    cv2.imshow("Stock Prediction", img_cv)

cv2.destroyAllWindows()