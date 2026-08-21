import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib

# 1. 載入 27 種真值表查表模組
from rule_table import get_final_decision

# 2. 載入自定義 PPG 模型類別
from models.ppg_model import StressNN

# 3. 載入異常監控與通報模組
from anomaly_monitor import AnomalyDetector, send_alert_notification


# ----------------------------------------------------
# 1. 預設 Placeholder 模型 (當組員尚未提供模型時使用)
# ----------------------------------------------------
class DummyEDAModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super(DummyEDAModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.classes = ["Normal", "Suspected_Arousal", "High_Arousal"]

    def forward(self, x):
        return self.net(x)

    def predict_state(self, eda_tensor):
        """推論並返回 EDA 三階段狀態"""
        with torch.no_grad():
            logits = self.forward(eda_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
        return self.classes[pred_idx]


class DummyIMUModel(nn.Module):
    def __init__(self, in_channels=6, num_classes=3):
        super(DummyIMUModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.classes = ["calm", "slightly_intense", "fierce"]

    def forward(self, x):
        return self.net(x)

    def predict_state(self, imu_raw, imu_tensor):
        """優先以物理 G 值加速度劃分 IMU 狀態，兼顧模型輸出"""
        acc_x, acc_y, acc_z = imu_raw[0, 0, :], imu_raw[0, 1, :], imu_raw[0, 2, :]
        max_g = np.max(np.sqrt(acc_x**2 + acc_y**2 + acc_z**2))
        
        if max_g < 1.2:
            return "calm"
        elif 1.2 <= max_g <= 2.5:
            return "slightly_intense"
        else:
            return "fierce"


# ----------------------------------------------------
# 2. 多模態壓力與異常檢測預測器類別
# ----------------------------------------------------
class MultimodalStressPredictor:
    """
    多模態 (PPG, EDA, IMU) 27 種真值表查表推論整合類別
    """

    def __init__(self, weights_dir="weights"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = weights_dir

        # ----------------------------------------------------
        # 載入 PPG 相關權重與預處理器 (你的部分)
        # ----------------------------------------------------
        ppg_scaler_path = os.path.join(weights_dir, "ppg_scaler.pkl")
        label_encoder_path = os.path.join(weights_dir, "label_encoder.pkl")
        ppg_weight_path = os.path.join(weights_dir, "ppg_stress_model.pth")

        if os.path.exists(ppg_scaler_path) and os.path.exists(label_encoder_path):
            self.ppg_scaler = joblib.load(ppg_scaler_path)
            self.label_encoder = joblib.load(label_encoder_path)
            self.num_classes = len(self.label_encoder.classes_)
            print("✓ 成功載入 ppg_scaler.pkl 與 label_encoder.pkl")
        else:
            print("⚠️ 未找到 ppg_scaler.pkl 或 label_encoder.pkl，推論時將跳過 Scaler 標準化。")
            self.ppg_scaler = None
            self.label_encoder = None
            self.num_classes = 3

        # 初始化 PPG 模型
        self.ppg_model = StressNN(input_dim=19, num_classes=self.num_classes).to(self.device)
        if os.path.exists(ppg_weight_path):
            self.ppg_model.load_state_dict(
                torch.load(ppg_weight_path, map_location=self.device)
            )
            print("✓ 成功載入 ppg_stress_model.pth 權重")
        else:
            print("⚠️ 未找到 ppg_stress_model.pth，使用未訓練的預設 PPG 權重進行測試。")
        self.ppg_model.eval()

        # ----------------------------------------------------
        # 載入 EDA 模型 (組員 2)
        # ----------------------------------------------------
        # TODO: 當組員 2 提供 eda_model.py 時，改為: from models.eda_model import EDAModel
        self.eda_model = DummyEDAModel(
            in_channels=1, num_classes=self.num_classes
        ).to(self.device)
        eda_weight_path = os.path.join(weights_dir, "eda_model.pth")
        if os.path.exists(eda_weight_path):
            self.eda_model.load_state_dict(
                torch.load(eda_weight_path, map_location=self.device)
            )
            print("✓ 成功載入 eda_model.pth 權重")
        else:
            print("ℹ️ 未找到 eda_model.pth，使用 Dummy EDA 模型。")
        self.eda_model.eval()

        # ----------------------------------------------------
        # 載入 IMU 模型 (組員 3)
        # ----------------------------------------------------
        # TODO: 當組員 3 提供 imu_model.py 時，改為: from models.imu_model import IMUModel
        self.imu_model = DummyIMUModel(
            in_channels=6, num_classes=self.num_classes
        ).to(self.device)
        imu_weight_path = os.path.join(weights_dir, "imu_model.pth")
        if os.path.exists(imu_weight_path):
            self.imu_model.load_state_dict(
                torch.load(imu_weight_path, map_location=self.device)
            )
            print("✓ 成功載入 imu_model.pth 權重")
        else:
            print("ℹ️ 未找到 imu_model.pth，使用 Dummy IMU 模型。")
        self.imu_model.eval()

    def get_ppg_state(self, ppg_raw):
        """將 PPG 的心率 (HR) 數值離散化為 Slow / Medium / Fast"""
        hr = ppg_raw[0][6]  # Index 6 為 HR 心率
        if hr < 60:
            return "Slow"
        elif 60 <= hr <= 100:
            return "Medium"
        else:
            return "Fast"

    def predict(self, ppg_raw, eda_raw, imu_raw):
        """
        三模態狀態映射與 27 種真值表查表推論
        :param ppg_raw: shape (1, 19) 未縮放的 19 項 HRV 特徵
        :param eda_raw: shape (1, 1, Seq_Len) EDA 原始波形數據
        :param imu_raw: shape (1, 6, Seq_Len) IMU 6 軸數據
        :return: 包含個別狀態與 27 種真值表查表結果的字典
        """
        # --- 1. 轉為 PyTorch Tensor 供模型運算 ---
        eda_tensor = torch.tensor(eda_raw, dtype=torch.float32).to(self.device)
        imu_tensor = torch.tensor(imu_raw, dtype=torch.float32).to(self.device)

        # --- 2. 取得 PPG, EDA, IMU 三個模態的離散狀態 ---
        ppg_state = self.get_ppg_state(ppg_raw)
        eda_state = self.eda_model.predict_state(eda_tensor)
        imu_state = self.imu_model.predict_state(imu_raw, imu_tensor)

        # --- 3. 呼叫 rule_table.py 進行 27 種組合對照 ---
        final_decision = get_final_decision(ppg_state, eda_state, imu_state)

        return {
            "ppg_state": ppg_state,
            "eda_state": eda_state,
            "imu_state": imu_state,
            "final_decision": final_decision
        }


# ----------------------------------------------------
# 3. 測試執行流程 (模擬正常與異常數據輸入)
# ----------------------------------------------------
if __name__ == "__main__":
    print("=== 初始化多模態壓力檢測預測器與異常監控器 ===")
    predictor = MultimodalStressPredictor(weights_dir="weights")
    detector = AnomalyDetector()

    print("\n----------------------------------------")
    print("【測試情境 A】：傳入正常數據 (心率 72 BPM, IMU 平穩)")
    print("----------------------------------------")
    normal_ppg = np.array([
        [
            800.5, 790.0, 45.2, 32.1, 20.5, 1.4,
            72.0,  # HR = 72.0 (Medium)
            15.2, 5.4, 3.1, 0.2, 1.01, 1.00, 0.05, 0.04, 0.02, 1.25, 2.9, 0.1
        ]
    ])
    normal_eda = np.random.randn(1, 1, 1000)
    normal_imu = np.random.randn(1, 6, 1000) * 0.1  # G 值最小化 (calm)

    res_normal = predictor.predict(normal_ppg, normal_eda, normal_imu)
    report_normal = detector.evaluate_status(res_normal)
    send_alert_notification(report_normal)

    print("\n----------------------------------------")
    print("【測試情境 B】：傳入異常數據 (心率 145 BPM, IMU 劇烈晃動)")
    print("----------------------------------------")
    anomaly_ppg = np.array([
        [
            800.5, 790.0, 45.2, 32.1, 20.5, 1.4,
            145.0,  # HR = 145.0 (Fast)
            15.2, 5.4, 3.1, 0.2, 1.01, 1.00, 0.05, 0.04, 0.02, 1.25, 2.9, 0.1
        ]
    ])
    anomaly_eda = np.random.randn(1, 1, 1000)
    anomaly_imu = np.random.randn(1, 6, 1000) * 3.5  # G 值超標 (fierce)

    res_anomaly = predictor.predict(anomaly_ppg, anomaly_eda, anomaly_imu)
    report_anomaly = detector.evaluate_status(res_anomaly)
    send_alert_notification(report_anomaly)