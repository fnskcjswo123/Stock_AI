import os
import torch
import torch.nn as nn
import onnx

# ----------------------------
# 1️⃣ 간단한 신경망 정의
# ----------------------------
class SimpleNet(nn.Module):
    def __init__(self, input_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)  # 출력 1개

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ----------------------------
# 2️⃣ 폴더 경로 설정
# ----------------------------
labels_folder = r".\labels"  # txt 파일들이 들어있는 폴더
models_folder = r".\models"  # ONNX 파일을 저장할 폴더

# models 폴더 없으면 생성
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

for filename in os.listdir(labels_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(labels_folder, filename)
        data = []
        labels = []

        # ----------------------------
        # 3️⃣ TXT 파일 읽기
        # ----------------------------
        with open(file_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                labels.append(parts[0])
                data.append(parts[1:])

        # ----------------------------
        # 4️⃣ 텐서 변환
        # ----------------------------
        X = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).view(-1,1)

        input_size = X.shape[1]
        model = SimpleNet(input_size=input_size)
        model.eval()  # ONNX 변환용

        # ----------------------------
        # 5️⃣ ONNX로 변환 (models 폴더에 저장)
        # ----------------------------
        dummy_input = torch.randn(1, input_size)
        onnx_filename = filename.replace(".txt", ".onnx")
        onnx_path = os.path.join(models_folder, onnx_filename)  # 변경된 경로

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        print(f"{filename} → {onnx_filename} 생성 완료! (models 폴더)")
