# stock_ai_honest_final.py
# 진짜 미래를 예측하는 정직한 주식 AI - 완성판
import sys
import json
import time
import msvcrt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import matplotlib.pyplot as plt

# ================== ASCII 아트 ==================
stock_ai_ascii_art = r"""
  /$$$$$$ /$$ /$$ /$$$$$$ /$$$$$$
 /$$__ $$ | $$ | $$ /$$__ $$|_ $$_/
| $$ \__//$$$$$$ /$$$$$$ /$$$$$$$| $$ /$$ | $$ \ $$ | $$
| $$$$$$|_ $$_/ /$$__ $$ /$$_____/| $$ /$$/ | $$$$$$$$ | $$
 \____ $$ | $$ | $$ \ $$| $$ | $$$$$$/ | $$__ $$ | $$
 /$$ \ $$ | $$ /$$| $$ | $$| $$ | $$_ $$ | $$ | $$ | $$
| $$$$$$/ | $$$$/| $$$$$$/| $$$$$$$| $$ \ $$ | $$ | $$ /$$$$$$
 \______/ \___/ \______/ \_______/|__/ \__/ |__/ |__/|______/
"""

o_ascii = r"""
  /$$$$$$
 /$$__ $$
| $$ \ $$
| $$ | $$
| $$ | $$
| $$ | $$
| $$$$$$/
 \______/
"""

x_ascii = r"""
 /$$ /$$
| $$ / $$
| $$/ $$/
 \ $$$$/
  >$$ $$
 /$$/\ $$
| $$ \ $$
|__/ |__/
"""

same_ascii = r"""
 /$$$$$$$$$
|_________/
 /$$$$$$$$$
|_________/
"""

print(stock_ai_ascii_art + "\n")

# ================== 경로 설정 ==================
def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = Path(sys.executable).parent
    else:
        base_path = Path(__file__).parent
    return base_path / relative_path

csv_parent = resource_path("csv")
config_path = resource_path("config")
settings_path = config_path / "settings.json"

# ================== 설정 파일 ==================
default_settings = {
    "graph": True,
    "future_days": 20,
    "trimmed_mean_percentage": 1
}

if not config_path.exists():
    config_path.mkdir(parents=True)
if not settings_path.exists():
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(default_settings, f, ensure_ascii=False, indent=4)

with open(settings_path, "r", encoding="utf-8") as f:
    settings = json.load(f)

graph = settings.get("graph", True)
future_days = settings.get("future_days", 20)
trim_pct = settings.get("trimmed_mean_percentage", 1)

# ================== CSV 폴더 선택 ==================
if not csv_parent.exists():
    print("'csv' 폴더가 없습니다!")
    time.sleep(5); sys.exit()

folders = [f for f in csv_parent.iterdir() if f.is_dir()]
if not folders:
    print("csv 안에 폴더가 하나도 없어요!")
    time.sleep(5); sys.exit()

if len(folders) == 1:
    selected = folders[0]
    print(f"자동 선택: {selected.name}")
else:
    print("사용 가능한 폴더:")
    for i, f in enumerate(folders, 1):
        print(f"{i}. {f.name}")
    while True:
        try:
            n = int(input("\n번호 입력: ")) - 1
            if 0 <= n < len(folders):
                selected = folders[n]
                break
        except:
            print("다시 입력하세요")

csv_files = list(selected.glob("*.csv"))
if not csv_files:
    print("CSV 파일이 없습니다!")
    time.sleep(5); sys.exit()

# ================== 데이터 로드 및 전처리 ==================
required_cols = ['종가', '시가', '고가', '저가', '거래량', '변동 %']
all_data = []

for f in csv_files:
    df = None
    for enc in ['utf-8', 'cp949']:
        try:
            df = pd.read_csv(f, encoding=enc, thousands=',')
            break
        except:
            continue
    if df is None or not all(c in df.columns for c in required_cols):
        continue

    df = df[required_cols].copy()
    df['거래량'] = pd.to_numeric(df['거래량'].astype(str).str.replace('M',''), errors='coerce').fillna(0) * 1e6
    df['변동 %'] = pd.to_numeric(df['변동 %'].astype(str).str.replace('%',''), errors='coerce').fillna(0) / 100
    for c in ['종가','시가','고가','저가']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    all_data.append(df.values[::-1])

raw_data = np.concatenate(all_data, axis=0)
print(f"총 {len(raw_data):,}일 데이터 로드 완료\n")

# 절사 평균 함수
def trimmed_array(arr, pct):
    if pct <= 0: return arr
    k = int(len(arr) * pct / 100)
    if k == 0: return arr
    sorted_arr = np.sort(arr)
    lower, upper = sorted_arr[k], sorted_arr[-k-1]
    return np.clip(arr, lower, upper)

# 정규화 (열 별 개별 스케일링)
scalers = [MinMaxScaler() for _ in range(6)]
data_scaled = np.zeros_like(raw_data, dtype=float)
for i in range(6):
    data_scaled[:, i] = scalers[i].fit_transform(raw_data[:, i].reshape(-1, 1)).ravel()

SEQ_LEN = 10
X = [data_scaled[i:i+SEQ_LEN] for i in range(len(data_scaled)-SEQ_LEN)]
y = [data_scaled[i+SEQ_LEN, 0] for i in range(len(data_scaled)-SEQ_LEN)]
actual_prices = raw_data[SEQ_LEN:, 0]

X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

# ================== 모델 정의 ==================
class GRUStock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(6, 64, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

model = GRUStock()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ================== 학습 ==================
print("AI 학습 중... (조금 걸려요)")
for epoch in range(1, 501):
    optimizer.zero_grad()
    pred = model(X_tensor)
    loss = criterion(pred, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch % 100 == 0) or (epoch == 500):
        print(f"Epoch {epoch}/500 - Loss: {loss.item():.6f}")
print("학습 완료!\n")

# ================== 과거 예측 (착시용) ==================
model.eval()
with torch.no_grad():
    past_pred_scaled = model(X_tensor).numpy().flatten()
    dummy = np.zeros((len(past_pred_scaled), 6))
    dummy[:, 0] = past_pred_scaled
    past_pred_prices = scalers[0].inverse_transform(dummy[:, [0]])[:, 0]
    past_pred_prices = trimmed_array(past_pred_prices, trim_pct)

# ================== 진짜 미래 예측 (autoregressive) ==================
print(f"진짜 {future_days}일 후 예측 시작...\n")
future_prices = []
last_seq = data_scaled[-SEQ_LEN:].copy()

with torch.no_grad():
    for day in range(1, future_days + 1):
        x_in = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)
        pred_scaled = model(x_in).item()

        dummy = np.zeros((1, 6))
        dummy[0, 0] = pred_scaled
        real_price = scalers[0].inverse_transform(dummy[:, [0]])[0, 0]
        future_prices.append(real_price)

        # 다음 입력으로 사용
        new_row = last_seq[-1].copy()
        new_row[0] = pred_scaled
        last_seq = np.roll(last_seq, -1, axis=0)
        last_seq[-1] = new_row

        trend = "상승" if real_price > actual_prices[-1] else "하락" if real_price < actual_prices[-1] else "횡보"
        print(f"Day +{day:2d} → {real_price:,.0f}원 ({trend})")

# ================== 최종 판단 ==================
current = actual_prices[-1]
final_future = future_prices[-1]
change_pct = (final_future / current - 1) * 100

print("\n" + "="*60)
print("                   최종 AI 판단")
print("="*60)

if change_pct > 5:
    print(o_ascii)
    print(f"20일 후 예측가: {final_future:,.0f}원 (+{change_pct:+.1f}%)")
    print("AI: 상승 예상... 근데 이건 거의 로또입니다")
elif change_pct < -5:
    print(x_ascii)
    print(f"20일 후 예측가: {final_future:,.0f}원 ({change_pct:+.1f}%)")
    print("AI: 하락 예상... 그래도 틀릴 확률 80% 이상")
else:
    print(same_ascii)
    print("AI: 별로 안 변할 듯. 그냥 두세요")

print("\n이 AI는 재미로만 즐기세요. 진짜 투자 손실 책임 안 집니다")
print("진심입니다")

# ================== 그래프 (정직하게) ==================
if graph:
    plt.figure(figsize=(15, 8))
    plt.plot(actual_prices, label="실제 가격", color="black", linewidth=2)
    plt.plot(range(SEQ_LEN, len(actual_prices)), past_pred_prices,
             color="green", alpha=0.8, label="과거 예측 (거의 완벽)")

    future_x = range(len(actual_prices), len(actual_prices) + future_days)
    for i, p in enumerate(future_prices):
        alpha = max(0.2, 1 - i*0.04)
        plt.plot(future_x[i], p, 'o', color='blue', alpha=alpha, markersize=9)
    plt.plot(future_x, future_prices, '--', color='blue', alpha=0.7, linewidth=2,
             label="진짜 미래 예측 (점점 불확실)")

    plt.axvline(len(actual_prices)-1, color='red', linestyle='--', linewidth=3)
    plt.text(len(actual_prices), plt.ylim()[1]*0.9, "여기서부터\n진짜 미래\n(불확실함)", 
             fontsize=14, color='red', ha='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="pink", alpha=0.8))

    plt.title("정직한 주식 예측 AI - 과거는 잘 맞지만 미래는...", fontsize=18)
    plt.xlabel("일자")
    plt.ylabel("종가 (원)")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\n아무 키나 누르면 종료됩니다...")
msvcrt.getch()