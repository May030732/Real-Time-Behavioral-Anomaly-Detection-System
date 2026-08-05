# 心率壓力檢測 AI 模型說明文件 (Model Documentation)

本文件整理了心率壓力檢測模型 (`StressNN`) 的架構紀錄、輸入特徵與標籤細節、模型權重儲存檔案，以及如何使用訓練好的模型進行推論的範例說明。

---

## 1. 目前模型紀錄的內容有哪些 (Model Records)

在訓練與評估過程中，系統主要紀錄了以下四大類別的內容：

1. **硬體與運算環境資訊：**
   - 自動偵測並記錄使用 **CUDA GPU** 或 **CPU** 執行運算。
2. **神經網路架構規格 (`StressNN`)：**
   - **輸入層 (Input Layer)：** 接受 `19` 個特徵維度。
   - **隱藏層 1 (Hidden Layer 1)：** `128` 個神經元 + 批次歸一化 (`BatchNorm1d`) + `ReLU` 激活函數 + `Dropout(0.3)`。
   - **隱藏層 2 (Hidden Layer 2)：** `64` 個神經元 + 批次歸一化 (`BatchNorm1d`) + `ReLU` 激活函數。
   - **輸出層 (Output Layer)：** `3` 個輸出類別的分數 (`num_classes = 3`)。
3. **訓練過程參數與指標：**
   - 訓練輪數 (`Epochs = 30`)、批次大小 (`Batch Size = 256`)。
   - 優化器 (`Adam, lr = 0.001`) 與 交叉熵損失函數 (`nn.CrossEntropyLoss()`) 計算出之 Loss 變化。
4. **模型評估結果 (Evaluation Metrics)：**
   - 記錄測試集上的 **Accuracy（準確率）**、**Precision（精確率）**、**Recall（召回率）** 與 **F1-Score**。

---

## 2. 特徵與對應的答案有哪些 (Features & Target)

### 2.1 輸入特徵 (19 項 HRV 時域特徵)
資料來自 `time_domain_features_train.csv`，包含以下 19 個經過標準化處理解的生理訊號指標：

| 特徵名稱 | 說明與生理意義 |
| :--- | :--- |
| `MEAN_RR` | 平均心跳間隔 (RR Interval) |
| `MEDIAN_RR` | 中位數心跳間隔 |
| `SDRR` | 心跳間隔標準差（整體變異度指標） |
| `RMSSD` | 相鄰心跳間隔差值的均方根（**副交感神經與放鬆程度核心指標**） |
| `SDSD` | 相鄰心跳間隔差值的標準差 |
| `SDRR_RMSSD` | SDRR 與 RMSSD 的比值 |
| `HR` | 平均心率 (Heart Rate, BPM) |
| `pNN25` | 相鄰心跳差值大於 25ms 的百分比 |
| `pNN50` | 相鄰心跳差值大於 50ms 的百分比 |
| `KURT` | 心跳間隔分佈之峰度 (Kurtosis) |
| `SKEW` | 心跳間隔分佈之偏度 (Skewness) |
| `MEAN_REL_RR` | 相對 RR 間期間隔平均值 |
| `MEDIAN_REL_RR` | 相對 RR 間期間隔中位數 |
| `SDRR_REL_RR` | 相對 RR 間期間隔標準差 |
| `RMSSD_REL_RR` | 相對 RR 間期間隔 RMSSD |
| `SDSD_REL_RR` | 相對 RR 間期間隔 SDSD |
| `SDRR_RMSSD_REL_RR` | 相對 RR 間期間隔之 SDRR/RMSSD 比值 |
| `KURT_REL_RR` | 相對 RR 間期間隔峰度 |
| `SKEW_REL_RR` | 相對 RR 間期間隔偏度 |

### 2.2 對應的真實答案 (Target / Condition)
答案儲存在 `heart_rate_non_linear_features_train.csv` 的 **`condition`** 欄位中，經 `LabelEncoder` 轉化為整數：

- **`0` (`interruption`)**：突發干擾 / 驚嚇狀態
- **`1` (`no stress`)**：無壓力 / 平靜狀態
- **`2` (`time pressure`)**：時間壓力 / 心理緊張狀態

---

## 3. 訓練好模型的儲存檔案 (Model Saved Artifacts)

完整訓練並部署模型需要 **「模型三件套」**：

1. **`stress_model.pth`**：
   - PyTorch 模型權重檔案，紀錄神經網路中所有 Linear 與 BatchNorm 層的參數。
2. **`scaler.pkl`**：
   - 特徵標準化工具 (`StandardScaler`)，儲存了訓練集的均值 (mean) 與標準差 (std)，**推論新資料時必須使用相同的參數進行縮放**。
3. **`label_encoder.pkl`**：
   - 標籤轉換工具 (`LabelEncoder`)，儲存 `0, 1, 2` 與 `interruption, no stress, time pressure` 的映射關係。

---

## 4. 如何使用訓練好的模型進行推論 (Inference / Prediction)

以下提供完整的推論腳本範例 `predict.py`，說明如何載入模型與權重並對新資料進行預測：

```python
import torch
import torch.nn as nn
import numpy as np
import joblib

# ----------------------------------------------------
# 1. 設置運算裝置 (Device Configuration)
# ----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------
# 2. 載入模型預處理器 (Scaler & Label Encoder)
# ----------------------------------------------------
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ----------------------------------------------------
# 3. 定義模型結構 (必須與訓練時的架構完全相同)
# ----------------------------------------------------
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

# 初始化模型並載入訓練好的權重 (.pth)
input_dim = scaler.mean_.shape[0]  # 19
num_classes = len(label_encoder.classes_)  # 3

model = StressNN(input_dim=input_dim, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("stress_model.pth", map_location=device))
model.eval()  # 切換至評估模式 (關閉 Dropout / 將 BatchNorm 設定為固定模式)

# ----------------------------------------------------
# 4. 輸入新資料並進行預測 (Prediction)
# ----------------------------------------------------
# 模擬輸入一筆包含 19 個 HRV 特徵的原始數據
sample_data = np.array([[
    800.5, 790.0, 45.2, 32.1, 20.5, 1.4, 75.0, 15.2, 5.4, 3.1, 0.2,
    1.01, 1.00, 0.05, 0.04, 0.02, 1.25, 2.9, 0.1
]])

# Step 1: 使用相同的 Scaler 進行標準化
scaled_data = scaler.transform(sample_data)

# Step 2: 轉換為 PyTorch Tensor 並送至對應裝置
input_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(device)

# Step 3: 前向傳播與 Softmax 計算機率
with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item() * 100

# Step 4: 將數字類別轉回文字標籤
predicted_label = label_encoder.inverse_transform([predicted_class])[0]

print(f"預測壓力狀態: {predicted_label}")
print(f"預測信心度: {confidence:.2f}%")
```
