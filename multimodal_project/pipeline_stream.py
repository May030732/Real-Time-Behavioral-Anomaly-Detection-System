import threading
import time
import numpy as np

import imu_test
from anomaly_monitor import send_alert_notification


class WristbandDataProcessor:
    def __init__(self, imu_window_size=120):
        self.imu_window_size = imu_window_size

    def extract_ppg_19_features(self, raw_ppg_wave):
        """
        將手環傳入的 PPG 原始波形轉換為 19 項 HRV 特徵。
        """
        hr_value = 75.0 + np.random.uniform(-5.0, 5.0)
        calculated_hrv = np.array([
            800.5, 790.0, 45.2, 32.1, 20.5, 1.4, hr_value, 15.2, 5.4, 3.1, 0.2,
            1.01, 1.00, 0.05, 0.04, 0.02, 1.25, 2.9, 0.1
        ])
        return calculated_hrv.reshape(1, 19)

    def get_latest_eda_value(self, raw_eda_stream):
        """
        隊友 EDADetector.detect_eda(current_eda) 接收單筆最新 EDA 數值 (float)。
        """
        return float(raw_eda_stream[-1])

    def format_imu_rows(self, raw_imu_acc_gyro):
        """
        隊友 imu_test.predict_imu_rows(rows) 接收 [[acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z], ...]
        預設長度為 120 筆 (60Hz * 2s)。
        """
        arr = np.asarray(raw_imu_acc_gyro, dtype=np.float32)
        if arr.shape[0] == 6 and arr.shape[1] != 6:
            arr = arr.T  # 轉為 (N, 6)
        return arr[-self.imu_window_size:].tolist()


class WearableStreamWorker:
    """
    背景執行緒手環串流處理器，與主程式 Camera 迴圈平行執行。
    """

    def __init__(self, predictor, detector, interval_sec=2.0):
        self.predictor = predictor
        self.detector = detector
        self.interval_sec = interval_sec

        imu_win = imu_test.get_model_info().get("window_size") or 120
        self.processor = WristbandDataProcessor(imu_window_size=imu_win)

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_state = {
            "ppg_state": "NORMAL",
            "eda_state": "Normal",
            "imu_state": "calm",
            "final_decision": "Normal",
        }

    def get_latest_state(self):
        with self._lock:
            return dict(self._latest_state)

    def stop(self):
        self._stop_event.set()

    def run(self):
        print("\n🚀 [背景執行緒] 開始接收手環數據串流與 36 種真值表查表監控...")
        imu_win = self.processor.imu_window_size

        while not self._stop_event.is_set():
            # TODO: 替換為真實 ESP32 串口/藍牙讀取的感測器緩衝區
            raw_ppg_stream = np.random.randn(1000)
            current_eda_val = float(np.random.uniform(0.3, 0.5))
            raw_imu_rows = np.random.randn(imu_win, 6) * 0.1

            ppg_19_feats = self.processor.extract_ppg_19_features(raw_ppg_stream)
            imu_rows = self.processor.format_imu_rows(raw_imu_rows)

            # 呼叫整合後的手環預測器 (PPG 4類 + 隊友 EDA + 隊友 IMU + 36規則表)
            prediction = self.predictor.predict(
                ppg_raw=ppg_19_feats,
                current_eda_value=current_eda_val,
                imu_rows=imu_rows,
            )

            with self._lock:
                self._latest_state = prediction

            report = self.detector.evaluate_status(prediction)
            send_alert_notification(report)

            self._stop_event.wait(self.interval_sec)

        print("已停止手環數據串流背景執行緒。")
