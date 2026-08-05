import torch
import torch.nn as nn
import numpy as np
import joblib

# ==========================================
# 1. 檢查硬體裝置 (GPU / CPU)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 目前推論使用的裝置: {device}")

# ==========================================
# 2. 載入預處理工具 (.pkl)
# ==========================================
try:
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print("✅ 成功載入 scaler.pkl 與 label_encoder.pkl")
except FileNotFoundError as e:
    print(f"❌ 找不到預處理工具檔案，請確認檔案路徑：\n{e}")
    exit()

# ==========================================
# 3. 定義與訓練時「完全相同」的神經網路架構
# ==========================================
class StressNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(StressNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 取得輸入欄位數量與答案類別數量
input_dim = scaler.mean_.shape[0]        # 從 scaler 自動抓取訓練時的特徵數量
num_classes = len(label_encoder.classes_)  # 抓取類別數量 (例如 3)

# 實例化模型並載入權重 (.pth)
model = StressNN(input_dim=input_dim, num_classes=num_classes).to(device)

try:
    model.load_state_dict(torch.load("stress_model.pth", map_location=device))
    model.eval()  # 切換為評估/推論模式 (會停用 Dropout 與 BatchNorm 的訓練行為)
    print("✅ 成功載入 stress_model.pth 權重！\n")
except FileNotFoundError as e:
    print(f"❌ 找不到權重檔案：\n{e}")
    exit()

# ==========================================
# 4. 模擬一筆來自感測器的全新心率數據
# ==========================================
# 這裡建立一筆與訓練欄位數相同的隨機測試資料 (1 列, input_dim 欄)
# 未來替換為 ESP32 或 PPG 算出的真實時域特徵值即可
dummy_raw_features = np.random.uniform(low=500, high=1000, size=(1, input_dim))

print(f"📊 1. 原始感測器特徵數據 (前 5 個數值): {dummy_raw_features[0][:5]}")

# 5. 使用載入的 scaler 進行特徵標準化 (Standardization)
scaled_features = scaler.transform(dummy_raw_features)
print(f"⚙️ 2. 標準化後的數據 (前 5 個數值): {scaled_features[0][:5]}")

# 6. 轉成 PyTorch Tensor 並推送到 GPU
input_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)

# ==========================================
# 5. 模型實時推論 (Inference)
# ==========================================
with torch.no_grad(): # 推論不需要計算梯度，節省記憶體與時間
    outputs = model(input_tensor)
    
    # 使用 Softmax 將模型的原始輸出 (Logits) 轉為 0% ~ 100% 的機率值
    probabilities = torch.softmax(outputs, dim=1)
    
    # 抓出機率最高的那一個類別數字 (例如 0, 1, 2 中的某一個)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item() * 100

# 7. 使用 label_encoder 將數字轉回人類看得懂的文字標籤
predicted_label = label_encoder.inverse_transform([predicted_class])[0]

# ==========================================
# 6. 印出最終預測結果
# ==========================================
print("\n" + "="*40)
print(f"🔮 【預測結果】: {predicted_label}")
print(f"🎯 【預測信心度】: {confidence:.2f}%")
print("="*40)

# 印出各類別詳細機率分佈
print("\n各類別的詳細機率:")
for idx, class_name in enumerate(label_encoder.classes_):
    prob = probabilities[0][idx].item() * 100
    print(f"  • {class_name}: {prob:.2f}%")