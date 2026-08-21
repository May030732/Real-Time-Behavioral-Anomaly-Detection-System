import time
import numpy as np
from main_inference import MultimodalStressPredictor
from anomaly_monitor import AnomalyDetector, send_alert_notification

# ----------------------------------------------------
# 1. 手環 raw 資料處理器 (Preprocess Engine)
# ----------------------------------------------------
class WristbandDataProcessor:
    def __init__(self):
        pass

    def extract_ppg_19_features(self, raw_ppg_wave):
        """
        將手環傳入的 PPG 原始波形轉換為 19 項 HRV 特徵
        (實務上會呼叫 HeartPy 或 NeuroKit2 進行峰值檢測與特徵計算)
        """
        # 模擬即時特徵提取，動態產生心率數值 (index 6: HR) 供 27 種真值表測試
        # 預設正常心率 75.0 BPM
        hr_value = 75.0 + np.random.uniform(-5.0, 5.0)
        
        calculated_hrv = np.array([
            800.5, 790.0, 45.2, 32.1, 20.5, 1.4, hr_value, 15.2, 5.4, 3.1, 0.2,
            1.01, 1.00, 0.05, 0.04, 0.02, 1.25, 2.9, 0.1
        ])
        return calculated_hrv.reshape(1, 19)

    def process_eda_wave(self, raw_eda_wave):
        """將 EDA 波形進行濾波並裁切為 (1, 1, 1000)"""
        return np.array(raw_eda_wave, dtype=np.float32).reshape(1, 1, -1)

    def process_imu_wave(self, raw_imu_acc_gyro):
        """將 IMU 6 軸數據 (AccXYZ + GyroXYZ) 格式化為 (1, 6, 1000)"""
        return np.array(raw_imu_acc_gyro, dtype=np.float32).reshape(1, 6, -1)


# ----------------------------------------------------
# 2. 即時串流資料結合與 27 種真值表查表通報主迴圈
# ----------------------------------------------------
def start_wristband_stream_monitoring():
    # 初始化處理器、預測器與異常監控器
    processor = WristbandDataProcessor()
    predictor = MultimodalStressPredictor(weights_dir="weights")
    detector = AnomalyDetector()

    print("\n🚀 開始接收手環即時數據串流與 27 種真值表查表監控 (按 Ctrl+C 結束)...")

    try:
        while True:
            # --- Step A: 模擬接收手環傳來的原始 Sensor 數據 ---
            # (未來替換為真實串口/藍牙讀取的原始陣列)
            raw_ppg_stream = np.random.randn(1000) 
            raw_eda_stream = np.random.randn(1000)
            raw_imu_stream = np.random.randn(6, 1000) * 0.1  # 預設動態較為平穩

            # --- Step B: 資料結合與預處理轉換 ---
            ppg_19_feats = processor.extract_ppg_19_features(raw_ppg_stream)
            eda_formatted = processor.process_eda_wave(raw_eda_stream)
            imu_formatted = processor.process_imu_wave(raw_imu_stream)

            # --- Step C: 多模態推論與 27 種組合查表 ---
            # 回傳包含 ppg_state, eda_state, imu_state, final_decision 的字典
            prediction = predictor.predict(ppg_19_feats, eda_formatted, imu_formatted)

            # --- Step D: 評估 27 種真值表查表結果並發布通報 ---
            report = detector.evaluate_status(prediction)
            send_alert_notification(report)

            # 模擬手環每 2 秒傳送一次滑動視窗數據
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n已停止手環數據串流監控。")

if __name__ == "__main__":
    start_wristband_stream_monitoring()