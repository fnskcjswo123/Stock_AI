import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import onnx
import onnxruntime as ort
from pathlib import Path

# -----------------------------
# 1️⃣ 폴더 내 모든 CSV 읽기
# -----------------------------
folder_path = Path(r".\csv")  # CSV 파일들이 들어있는 폴더
all_data = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(folder_path, filename))
        # 거래량 숫자로 변환
        df['거래량'] = df['거래량'].str.replace('M','').astype(float) * 1e6
        # 변동 % 숫자로 변환
        df['변동 %'] = df['변동 %'].str.replace('%','').astype(float) / 100
        # 필요한 컬럼 선택 및 순서 뒤집기 (과거 -> 현재)
        data = df[['종가','시가','고가','저가','거래량','변동 %']].values[::-1]
        all_data.append(data)

# 모든 CSV 데이터 연결
all_data = np.concatenate(all_data, axis=0)

# 정규화
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(all_data)

# -----------------------------
# 2️⃣ 시퀀스 생성
# -----------------------------
SEQ_LENGTH = 10
X_list, y_list = [], []

for i in range(len(data_scaled)-SEQ_LENGTH):
    X_list.append(data_scaled[i:i+SEQ_LENGTH])
    y_list.append(data_scaled[i+SEQ_LENGTH][0])  # 종가 예측

X = torch.tensor(np.array(X_list), dtype=torch.float32)
y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)

# -----------------------------
# 3️⃣ GRU 모델 정의
# -----------------------------
class GRUStock(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, num_layers=1):
        super(GRUStock, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

model = GRUStock()

# -----------------------------
# 4️⃣ 학습
# -----------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

# -----------------------------
# 5️⃣ 예측 및 시각화
# -----------------------------
model.eval()
with torch.no_grad():
    pred = model(X).numpy()
    pred_prices = scaler.inverse_transform(
        np.concatenate([pred, np.zeros((pred.shape[0],5))], axis=1)
    )[:,0]
    actual_prices = all_data[SEQ_LENGTH:,0]  # 실제 종가

plt.plot(actual_prices, label='Actual')
plt.plot(pred_prices, label='Predicted')
plt.legend()
plt.show()

# -----------------------------
# 6️⃣ ONNX 변환
# -----------------------------
dummy_input = torch.randn(1, SEQ_LENGTH, 6)
torch.onnx.export(model, dummy_input, "gru_stock.onnx",
                  input_names=['input'], output_names=['output'])
print("ONNX 모델 저장 완료!")

# -----------------------------
# 7️⃣ ONNX로 예측 예제
# -----------------------------
ort_sess = ort.InferenceSession("gru_stock.onnx")
ort_pred = ort_sess.run(None, {"input": dummy_input.numpy()})
print("ONNX 모델 예측 예시:", ort_pred[0])
