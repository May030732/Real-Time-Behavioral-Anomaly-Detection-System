import os
import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from models.ppg_model import StressNN

# 1. 確保 weights 資料夾存在
os.makedirs("weights", exist_ok=True)

print("正在為你的 PPG 模型產生預設權重與轉換檔...")

# 2. 模擬產生 19 項 HRV 特徵數據與 3 種壓力狀態標籤
np.random.seed(42)
dummy_x = np.random.randn(100, 19)
dummy_y = np.array(["interruption", "no stress", "time pressure"] * 33 + ["no stress"])

# 3. 匯出 ppg_scaler.pkl
scaler = StandardScaler()
scaler.fit(dummy_x)
joblib.dump(scaler, "weights/ppg_scaler.pkl")
print("✓ 已成功產生 weights/ppg_scaler.pkl")

# 4. 匯出 label_encoder.pkl
label_encoder = LabelEncoder()
label_encoder.fit(dummy_y)
joblib.dump(label_encoder, "weights/label_encoder.pkl")
print("✓ 已成功產生 weights/label_encoder.pkl")

# 5. 初始化你的 StressNN 模型並匯出 ppg_stress_model.pth
model = StressNN(input_dim=19, num_classes=3)
torch.save(model.state_dict(), "weights/ppg_stress_model.pth")
print("✓ 已成功產生 weights/ppg_stress_model.pth")

print("\n🎉 完成！現在你的 weights/ 資料夾底下已經有這 3 個檔案了。")