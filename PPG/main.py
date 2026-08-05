import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. 檢查並設定 CUDA 硬體加速
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 50)
print(f"🚀 目前使用的運算裝置: {device}")
if device.type == "cuda":
    print(f"🎮 顯示卡型號: {torch.cuda.get_device_name(0)}")
print("=" * 50)

# ==========================================
# 2. 讀取並合併資料集 (解決找不到 condition 的問題)
# ==========================================
# 設定檔案路徑
base_dir = os.path.join("Heart_Rate_Stress", "Train Data", "Train Data Zip")
time_file = os.path.join(base_dir, "time_domain_features_train.csv")
nonlinear_file = os.path.join(base_dir, "heart_rate_non_linear_features_train.csv")

try:
    print("正在讀取並合併資料集 ...")
    # 讀取兩個 CSV 檔
    df_time = pd.read_csv(time_file)
    df_nonlinear = pd.read_csv(nonlinear_file)
    
    # 透過 'uuid' 將時域特徵與包含標籤(condition)的表格合併起來
    df = pd.merge(df_time, df_nonlinear[['uuid', 'condition']], on='uuid')
    
    print("【成功】資料合併成功！包含的目標標籤類別有：", df['condition'].unique())
    print("\n" + "="*50 + "\n")

    target_col = 'condition'
    
    # 提取特徵 (X) 與 答案標籤 (y)
    X = df.drop(columns=[target_col, 'uuid'], errors='ignore').values
    y = df[target_col].values

    # 將文字標籤 (例如 'no stress') 轉為 AI 看得懂的數字 (0, 1, 2)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    # 拆分訓練集與驗證集 (80% 訓練 / 20% 測試)
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 特徵標準化 (神經網路必備步驟，讓數值在相近的範圍內)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 轉為 PyTorch 張量 (Tensor) 並搬到 CUDA (GPU) 上
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    # 建立 DataLoader（小批次處理，加速 GPU 運算）
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # ==========================================
    # 3. 定義 PyTorch 神經網路模型 (MLP)
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

    # 實例化模型並推送到 GPU
    model = StressNN(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==========================================
    # 4. 開始在 GPU 上訓練模型
    # ==========================================
    epochs = 30
    print(f"🔥 開始進行 {epochs} 輪的 CUDA 神經網路訓練...\n")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")

    # ==========================================
    # 5. 評估模型效能
    # ==========================================
    model.eval()
    with torch.no_grad():
        outputs = model(X_val_tensor)
        _, predictions = torch.max(outputs, 1)
        
        # 把 CUDA 張量轉回 CPU 以便印出 Scikit-learn 的報告
        y_val_cpu = y_val_tensor.cpu().numpy()
        preds_cpu = predictions.cpu().numpy()

    print("\n🎉【訓練完成！】PyTorch + CUDA 模型評估報告如下：")
    print(classification_report(y_val_cpu, preds_cpu, target_names=label_encoder.classes_))

    # ==========================================
    # 6. 儲存模型與預處理工具
    # ==========================================
    import joblib

    # 1. 儲存 PyTorch 神經網路權重
    torch.save(model.state_dict(), "stress_model.pth")
    
    # 2. 儲存標準化工具 (Scaler) 與標籤轉換器 (LabelEncoder)
    # ⚠️ 這兩個非常重要！未來新資料必須用相同的參數縮放與解碼
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")

    print("\n💾 【成功】模型與預處理工具已順利儲存至本地資料夾！")
    print(" ├── stress_model.pth (神經網路權重)")
    print(" ├── scaler.pkl (特徵標準化工具)")
    print(" └── label_encoder.pkl (標籤編碼器)")

except FileNotFoundError as e:
    print(f"❌ 找不到檔案！請檢查路徑是否有誤：\n{e}")
except Exception as e:
    print(f"❌ 發生錯誤：{e}")
    