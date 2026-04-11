import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# 1️⃣ CSV 파일 읽기 및 전처리 함수
# -----------------------------
def read_and_preprocess_csvs(folder_path: Path):
    """지정된 폴더의 모든 CSV를 읽고 전처리합니다."""
    required_cols = ['종가', '시가', '고가', '저가', '거래량', '변동 %']
    all_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = folder_path / filename
            
            # 여러 인코딩을 순차적으로 시도하여 파일을 읽습니다.
            encodings_to_try = ['utf-8', 'cp949']
            df = None
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, thousands=',')
                    print(f"[{filename}] {encoding}로 성공적으로 읽었습니다.")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"[{filename}] {encoding} 시도 중 예상치 못한 오류 발생: {e}")
                    
            if df is None:
                print(f"[건너뜀] {filename}: 지원하는 인코딩으로 읽을 수 없습니다.")
                continue

            # 컬럼 이름 앞뒤 공백 제거
            df.columns = df.columns.str.strip()
            
            try:
                # 필요한 컬럼이 모두 있는지 확인
                if all(col in df.columns for col in required_cols):
                    # 데이터 전처리
                    df['거래량'] = df['거래량'].astype(str).str.replace('M', '', regex=False).str.strip()
                    df['변동 %'] = df['변동 %'].astype(str).str.replace('%', '', regex=False).str.strip()
                    
                    df['거래량'] = pd.to_numeric(df['거래량'], errors='coerce').fillna(0) * 1e6
                    df['변동 %'] = pd.to_numeric(df['변동 %'], errors='coerce').fillna(0) / 100
                    
                    for col in ['종가', '시가', '고가', '저가']:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
                    data = df[required_cols].values[::-1]
                    all_data.append(data)
                else:
                    print(f"[건너뜀] {filename}: 필요한 컬럼이 모두 없습니다.")
                    print(f"** 파일의 실제 컬럼: {df.columns.tolist()}")
                    print(f"** 필요한 컬럼: {required_cols}")

            except Exception as e:
                print(f"[에러] {filename} 전처리 중 오류: {e}")
                
    if not all_data:
        raise ValueError("폴더 안에 올바른 CSV 파일이 없습니다!")
    
    return np.concatenate(all_data, axis=0)

# -----------------------------
# 2️⃣ 데이터 준비 및 실행 (이하 동일)
# -----------------------------
try:
    all_data = read_and_preprocess_csvs(Path("hccsv"))
except ValueError as e:
    print(e)
    sys.exit(1)
    
# -----------------------------
# 2️⃣ 데이터 준비
# -----------------------------
try:
    all_data = read_and_preprocess_csvs(Path("hccsv"))
except ValueError as e:
    print(e)
    sys.exit(1)

# 정규화
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(all_data)

# 시퀀스 생성
SEQ_LENGTH = 10
X_list, y_list = [], []

for i in range(len(data_scaled) - SEQ_LENGTH):
    X_list.append(data_scaled[i : i + SEQ_LENGTH])
    y_list.append(data_scaled[i + SEQ_LENGTH][0])  # 종가 예측

X = torch.tensor(np.array(X_list), dtype=torch.float32)
y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)

# -----------------------------
# 3️⃣ GRU 모델 정의 및 학습
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
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 50

print("모델 학습 시작...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.6f}")

# -----------------------------
# 4️⃣ 예측 및 시각화
# -----------------------------
model.eval()
with torch.no_grad():
    pred = model(X).numpy()
    pred_prices = scaler.inverse_transform(
        np.concatenate([pred, np.zeros((pred.shape[0], 5))], axis=1)
    )[:, 0]
    actual_prices = all_data[SEQ_LENGTH:, 0]  # 실제 종가

plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(pred_prices, label='Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 5️⃣ ONNX 변환 및 테스트
# -----------------------------
dummy_input = torch.randn(1, SEQ_LENGTH, 6)
onnx_path = model_folder_path / onnx_file_name
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path,
    input_names=['input'], 
    output_names=['output'],
    export_params=True,
    opset_version=11, # onnxruntime과 호환성 위해 opset 버전 지정
    do_constant_folding=True,
    dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}}
)
print(f"ONNX 모델 저장 완료! 경로: {onnx_path}")

try:
    # ONNX 모델 테스트
    ort_sess = ort.InferenceSession(str(onnx_path))
    # ONNX 모델은 Numpy 배열을 입력으로 받음
    ort_pred = ort_sess.run(None, {"input": dummy_input.numpy()})
    print("ONNX 모델 예측 예시:", ort_pred[0])
except Exception as e:
    print(f"ONNX 모델 테스트 중 오류 발생: {e}")