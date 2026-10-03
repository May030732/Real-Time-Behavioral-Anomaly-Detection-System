import csv
import math
import os
import sqlite3
import statistics
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

# ============================================================
# 1. 系統設定與路徑
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# 讀取來源：原始感測器資料庫
SOURCE_DB_PATH = (BASE_DIR / ".." / "anomaly_detection.db").resolve()

# 輸出目標：建立新的資料庫存取結果
RESULT_DB_PATH = (BASE_DIR / ".." / "detection_result.db").resolve()

# 輸出目標：CSV 紀錄檔
OUTPUT_CSV_PATH = (BASE_DIR / ".." / "fusion_output_log.csv").resolve()

# IMU 模型設定
MODEL_FILE = BASE_DIR / "model" / "layout25_intensity_random_forest_v1.joblib"
IMU_WINDOW_SIZE = 120  # 60Hz * 2s
VALID_IMU_LABELS = {"calm", "slightly_intense", "fierce"}

# EDA 參數
EDA_BASELINE_SAMPLE_COUNT = 30
EDA_SUSPECTED_THRESHOLD = 0.20
EDA_HIGH_THRESHOLD = 0.50
EDA_REQUIRED_ABNORMAL_COUNT = 3

# 新版 PPG 參數 (對應 15 秒視窗與三層檢測)
PPG_WINDOW_SIZE_SEC = 15          # 視窗長度 15 秒
PPG_EXTREME_LOW = 40
PPG_EXTREME_HIGH = 180
PPG_REQUIRED_EXTREME_WINDOWS = 2  # 連續 2 個視窗 (30秒)
PPG_SPIKE_DROP_RATIO = 0.4        # 與前 2 分鐘均值差距 40%
PPG_MIN_2MIN_BUFFER = 4           # 至少累積 4 個視窗 (1分鐘)
PPG_MAX_2MIN_BUFFER = 8           # 最多累積 8 個視窗 (2分鐘)
PPG_PERSISTENT_HIGH = 100
PPG_REQUIRED_HIGH_WINDOWS = 4     # 連續 4 個視窗 (1分鐘)

last_processed_imu_id = 0


# ============================================================
# 2. EDA 檢測模組
# ============================================================
class EDAModule:
    def __init__(self):
        self.baseline_values = []
        self.baseline_eda = None
        self.suspected_count = 0
        self.high_count = 0

    def process_gsr(self, current_eda):
        if current_eda is None or current_eda <= 0:
            return "Invalid_EDA", current_eda, self.baseline_eda

        if self.baseline_eda is None:
            self.baseline_values.append(current_eda)
            if len(self.baseline_values) >= EDA_BASELINE_SAMPLE_COUNT:
                self.baseline_eda = sum(self.baseline_values) / len(self.baseline_values)
            return "Collecting_Baseline", current_eda, self.baseline_eda

        eda_change = current_eda - self.baseline_eda
        change_ratio = eda_change / self.baseline_eda

        if change_ratio >= EDA_HIGH_THRESHOLD:
            self.high_count += 1
            self.suspected_count = 0
            status = "High_Arousal" if self.high_count >= EDA_REQUIRED_ABNORMAL_COUNT else "Normal"
        elif change_ratio >= EDA_SUSPECTED_THRESHOLD:
            self.suspected_count += 1
            self.high_count = 0
            status = "Suspected_Arousal" if self.suspected_count >= EDA_REQUIRED_ABNORMAL_COUNT else "Normal"
        else:
            self.suspected_count = 0
            self.high_count = 0
            status = "Normal"

        return status, current_eda, self.baseline_eda


# ============================================================
# 3. PPG 三層視窗檢測模組 (全新重構)
# ============================================================
class PPGModule:
    def __init__(self):
        self.current_window_samples = []
        self.history_2min_windows = deque(maxlen=PPG_MAX_2MIN_BUFFER)
        self.extreme_status_history = deque(maxlen=PPG_REQUIRED_EXTREME_WINDOWS)
        self.high_bpm_history = deque(maxlen=PPG_REQUIRED_HIGH_WINDOWS)
        self.last_sample_time = 0
        self.current_status = "Collecting_Window"
        self.last_evaluated_avg_bpm = 0.0

    def process_bpm(self, current_bpm):
        """
        接收感測器即時 BPM。
        每秒抽樣 1 次進入 15 秒視窗，未滿 15 秒沿用前一次狀態。
        """
        if current_bpm is None or current_bpm <= 0:
            return "Invalid_BPM", current_bpm, self.last_evaluated_avg_bpm

        now = time.time()
        # 控制每 1 秒抽取 1 筆數據，使 15 筆能真實代表 15 秒
        if now - self.last_sample_time >= 1.0:
            self.last_sample_time = now
            self.current_window_samples.append(current_bpm)

            # 當湊齊 15 秒視窗資料時進行三層評估
            if len(self.current_window_samples) >= PPG_WINDOW_SIZE_SEC:
                window_avg_bpm = sum(self.current_window_samples) / len(self.current_window_samples)
                self.current_window_samples.clear()
                self.last_evaluated_avg_bpm = window_avg_bpm
                self.current_status = self._evaluate_window(window_avg_bpm)

        return self.current_status, current_bpm, self.last_evaluated_avg_bpm

    def _evaluate_window(self, window_bpm):
        # [第一層：極限值判斷 (<40 或 >180，持續 30 秒)]
        is_extreme = (window_bpm < PPG_EXTREME_LOW) or (window_bpm > PPG_EXTREME_HIGH)
        self.extreme_status_history.append(is_extreme)

        if len(self.extreme_status_history) == PPG_REQUIRED_EXTREME_WINDOWS and all(self.extreme_status_history):
            self.history_2min_windows.append(window_bpm)
            return "CRITICAL_EXTREME_HR"

        # [第二層：突發性劇變 (與 2 分鐘均值差距 40% 以上)]
        if len(self.history_2min_windows) >= PPG_MIN_2MIN_BUFFER:
            baseline_avg = sum(self.history_2min_windows) / len(self.history_2min_windows)
            upper_bound = baseline_avg * (1 + PPG_SPIKE_DROP_RATIO)
            lower_bound = baseline_avg * (1 - PPG_SPIKE_DROP_RATIO)

            if window_bpm >= upper_bound or window_bpm <= lower_bound:
                self.history_2min_windows.append(window_bpm)
                return "SUDDEN_HR_CHANGE"

        # [第三層：持續性偏高 (>100，持續 1 分鐘)]
        is_high = window_bpm > PPG_PERSISTENT_HIGH
        self.high_bpm_history.append(is_high)

        if len(self.high_bpm_history) == PPG_REQUIRED_HIGH_WINDOWS and all(self.high_bpm_history):
            status = "PERSISTENT_HIGH_HR"
        else:
            status = "NORMAL"

        self.history_2min_windows.append(window_bpm)
        return status


# ============================================================
# 4. IMU 特徵計算與推論模組
# ============================================================
class IMUModule:
    def __init__(self, model_file=MODEL_FILE):
        if not model_file.exists():
            raise FileNotFoundError(f"找不到 IMU 模型檔：\n{model_file}")

        package = joblib.load(model_file)
        if not isinstance(package, dict):
            raise ValueError("模型檔格式不正確：預期 joblib 內容為 dictionary。")

        self.model = package["model"]
        self.feature_names = package["feature_names"]
        self.window_size = int(package.get("window_size", IMU_WINDOW_SIZE))

    @staticmethod
    def _calculate_signal_features(values):
        values = [float(v) for v in values]
        count = len(values)
        if count == 0:
            raise ValueError("IMU 訊號不可為空。")

        mean_val = statistics.fmean(values)
        std_val = statistics.stdev(values) if count > 1 else 0.0
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        squared_sum = sum(v * v for v in values)
        rms_val = math.sqrt(squared_sum / count)
        energy_val = squared_sum / count

        return {
            "mean": mean_val, "std": std_val, "min": min_val,
            "max": max_val, "range": range_val, "rms": rms_val, "energy": energy_val
        }

    def predict_imu_rows(self, rows):
        if len(rows) != self.window_size:
            raise ValueError(f"IMU 資料筆數 ({len(rows)}) 與預期 ({self.window_size}) 不符。")

        acc_x = [float(r[0]) for r in rows]
        acc_y = [float(r[1]) for r in rows]
        acc_z = [float(r[2]) for r in rows]
        gyr_x = [float(r[3]) for r in rows]
        gyr_y = [float(r[4]) for r in rows]
        gyr_z = [float(r[5]) for r in rows]

        acc_mag = [math.sqrt(x * x + y * y + z * z) for x, y, z in zip(acc_x, acc_y, acc_z)]
        gyr_mag = [math.sqrt(x * x + y * y + z * z) for x, y, z in zip(gyr_x, gyr_y, gyr_z)]

        signals = {
            "acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z, "acc_mag": acc_mag,
            "gyr_x": gyr_x, "gyr_y": gyr_y, "gyr_z": gyr_z, "gyr_mag": gyr_mag,
        }

        signal_order = ["acc_x", "acc_y", "acc_z", "acc_mag", "gyr_x", "gyr_y", "gyr_z", "gyr_mag"]
        feature_dict = {}

        for signal_name in signal_order:
            feats = self._calculate_signal_features(signals[signal_name])
            for feat_name, feat_val in feats.items():
                feature_dict[f"{signal_name}_{feat_name}"] = feat_val

        feature_vector = [float(feature_dict[name]) for name in self.feature_names]
        x_in = np.asarray([feature_vector], dtype=np.float32)
        prediction = str(self.model.predict(x_in)[0])

        if prediction not in VALID_IMU_LABELS:
            raise ValueError(f"模型輸出了未知狀態：{prediction}")

        return prediction


# ============================================================
# 5. 資料庫存取與決策融合輔助函式
# ============================================================
def init_result_database():
    """在 detection_result.db 建立 results 資料表"""
    with sqlite3.connect(str(RESULT_DB_PATH), timeout=5) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                latest_sensor_id INTEGER,
                imu_status TEXT,
                ppg_bpm REAL,
                ppg_status TEXT,
                eda_value REAL,
                eda_status TEXT,
                final_decision TEXT
            )
        """)
        conn.commit()


def fetch_sensor_data_window():
    """從 anomaly_detection.db 讀取 IMU 視窗與最新 PPG/EDA 數據"""
    global last_processed_imu_id

    if not SOURCE_DB_PATH.exists():
        return None

    try:
        with sqlite3.connect(str(SOURCE_DB_PATH), timeout=5) as conn:
            cursor = conn.cursor()

            # 抓取 IMU 視窗數據 (120 筆)
            cursor.execute("""
                SELECT id, ax, ay, az, gx, gy, gz 
                FROM sensor_data 
                WHERE id > ? 
                ORDER BY id ASC 
                LIMIT ?
            """, (last_processed_imu_id, IMU_WINDOW_SIZE))

            imu_rows = cursor.fetchall()

            if len(imu_rows) < IMU_WINDOW_SIZE:
                return None

            latest_imu_id = imu_rows[-1][0]
            imu_signal_data = [row[1:] for row in imu_rows]

            # 抓取最新 1 筆 PPG (bpm) 與 GSR (gsr_smooth)
            cursor.execute("""
                SELECT bpm, gsr_smooth 
                FROM sensor_data 
                ORDER BY id DESC 
                LIMIT 1
            """)
            latest_single_row = cursor.fetchone()

            bpm_value = latest_single_row[0] if latest_single_row else 0.0
            gsr_value = latest_single_row[1] if latest_single_row else 0.0

            return {
                "latest_imu_id": latest_imu_id,
                "imu_window": imu_signal_data,
                "bpm": bpm_value,
                "gsr_smooth": gsr_value
            }
    except sqlite3.OperationalError as e:
        print(f"[SQL Busy] 讀取原始資料庫忙線: {e}")
        return None


def evaluate_fusion_decision(imu_st, ppg_st, eda_st):
    """
    多感測器融合決策 (Decision Fusion Logic)
    對應新版 PPG 狀態: CRITICAL_EXTREME_HR, SUDDEN_HR_CHANGE, PERSISTENT_HIGH_HR, NORMAL
    """
    # 1. 最危急狀況：極端心率
    if ppg_st == "CRITICAL_EXTREME_HR":
        return "Medical_Emergency"

    # 2. 判斷生理指標是否處於激發/異常狀態
    is_ppg_aroused = ppg_st in ("SUDDEN_HR_CHANGE", "PERSISTENT_HIGH_HR")
    is_eda_aroused = eda_st in ("Suspected_Arousal", "High_Arousal")
    is_physiological_aroused = is_ppg_aroused or is_eda_aroused

    # 3. 結合 IMU 運動強度交叉驗證
    if imu_st in ("calm", "slightly_intense") and is_physiological_aroused:
        return "Emotional_Stress"
    elif imu_st == "fierce" and is_physiological_aroused:
        return "Physical_Exertion"
    elif imu_st == "fierce":
        return "High_Motion_Normal"
    elif any("Collecting" in str(s) for s in (imu_st, ppg_st, eda_st)):
        return "Calibrating"
    elif any("Invalid" in str(s) for s in (imu_st, ppg_st, eda_st)):
        return "Sensor_Warning"
    else:
        return "Normal"


def save_detection_result(timestamp_str, latest_id, imu_st, ppg_val, ppg_st, eda_val, eda_st, decision):
    """將判定結果寫入 detection_result.db"""
    try:
        with sqlite3.connect(str(RESULT_DB_PATH), timeout=5) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO results (
                    timestamp, latest_sensor_id, imu_status, 
                    ppg_bpm, ppg_status, eda_value, eda_status, final_decision
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp_str, latest_id, imu_st, ppg_val, ppg_st, eda_val, eda_st, decision))
            conn.commit()
    except sqlite3.OperationalError as e:
        print(f"[Save Warning] 寫入 detection_result.db 失敗: {e}")


# ==========================================
# 6. 主程式
# ==========================================
def main():
    global last_processed_imu_id

    print("=" * 70)
    print("🚀 多感測器即時融合推論系統 (IMU + PPG 3-Layer + EDA) 啟動...")
    print(f"原始資料庫 (讀取): {SOURCE_DB_PATH}")
    print(f"結果資料庫 (寫入): {RESULT_DB_PATH}")
    print(f"結果 CSV   (寫入): {OUTPUT_CSV_PATH}")
    print("=" * 70)

    # 建立新資料庫及結果表
    init_result_database()

    eda_module = EDAModule()
    ppg_module = PPGModule()
    imu_module = IMUModule()

    # 初始化 CSV 檔案
    csv_headers = [
        "timestamp", "latest_sensor_id",
        "imu_status", "ppg_bpm", "ppg_status",
        "eda_value", "eda_status", "final_decision"
    ]
    write_header = not OUTPUT_CSV_PATH.exists()
    csv_file = open(str(OUTPUT_CSV_PATH), mode="a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
    if write_header:
        csv_writer.writeheader()
        csv_file.flush()

    try:
        while True:
            data = fetch_sensor_data_window()

            if data is not None:
                # 1. 各感測器獨立推論
                imu_status = imu_module.predict_imu_rows(data["imu_window"])
                ppg_status, ppg_val, _ = ppg_module.process_bpm(data["bpm"])
                eda_status, eda_val, _ = eda_module.process_gsr(data["gsr_smooth"])

                # 2. 多模態綜合判定
                final_decision = evaluate_fusion_decision(imu_status, ppg_status, eda_status)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 3. 儲存至 detection_result.db
                save_detection_result(
                    current_time, data["latest_imu_id"],
                    imu_status, ppg_val, ppg_status,
                    eda_val, eda_status, final_decision
                )

                # 4. 儲存至 CSV
                csv_writer.writerow({
                    "timestamp": current_time,
                    "latest_sensor_id": data["latest_imu_id"],
                    "imu_status": imu_status,
                    "ppg_bpm": ppg_val,
                    "ppg_status": ppg_status,
                    "eda_value": eda_val,
                    "eda_status": eda_status,
                    "final_decision": final_decision
                })
                csv_file.flush()

                # 5. 更新索引與終端機輸出
                last_processed_imu_id = data["latest_imu_id"]
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] ID:{last_processed_imu_id:<6} | "
                    f"IMU: {imu_status:<15} | PPG: {ppg_status:<19} | "
                    f"EDA: {eda_status:<16} => 最終判定: {final_decision}"
                )

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[System] 已手動停止推論系統。")
    finally:
        csv_file.close()


if __name__ == "__main__":
    main()