import serial
import sqlite3
import time
from collections import deque

# ==================== 1. 設定參數 ====================
DB_NAME = "anomaly_detection.db"
COM_PORT = 'COM3'        # 請修改為您電腦對應的 COM Port 號碼
BAUD_RATE = 115200

# ----------------- 三層演算法門檻設定 -----------------
WINDOW_SIZE_SEC = 15          # 每 15 秒算一次平均 BPM

# 第一層：極限值
EXTREME_LOW_BPM = 40
EXTREME_HIGH_BPM = 180
REQUIRED_EXTREME_WINDOWS = 2  # 連續 2 個視窗 (30秒)

# 第二層：突發性劇變
SPIKE_DROP_RATIO = 0.4        # 比前 2 分鐘平均值高/低 40%
MIN_2MIN_BUFFER_COUNT = 4     # 至少需要 4 個視窗 (1分鐘)
MAX_2MIN_BUFFER_COUNT = 8     # 最多保留 8 個視窗 (2分鐘)

# 第三層：持續性偏高
PERSISTENT_HIGH_BPM = 100
REQUIRED_HIGH_WINDOWS = 4     # 連續 4 個視窗 (1分鐘)

# ----------------- 演算法狀態隊列 -----------------
current_15s_window = []
history_2min_windows = deque(maxlen=MAX_2MIN_BUFFER_COUNT)
extreme_status_history = deque(maxlen=REQUIRED_EXTREME_WINDOWS)
high_bpm_history = deque(maxlen=REQUIRED_HIGH_WINDOWS)
last_window_sample_time = 0   # 控制 1 秒抽樣一次 BPM 進入 15s 視窗


# ==================== 2. 三層異常檢測邏輯 ====================
def evaluate_window_bpm(window_bpm, cursor, conn):
    """依據三層檢測邏輯評估，並將分析結果寫入 anomaly_logs 資料表"""
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')

    # [第一層：極限值判斷]
    is_extreme = (window_bpm < EXTREME_LOW_BPM) or (window_bpm > EXTREME_HIGH_BPM)
    extreme_status_history.append(is_extreme)

    if len(extreme_status_history) == REQUIRED_EXTREME_WINDOWS and all(extreme_status_history):
        status_code = "CRITICAL_EXTREME_HR"
        history_2min_windows.append(window_bpm)
        _save_anomaly_log(cursor, conn, current_time, window_bpm, status_code)
        return status_code

    # [第二層：突發性劇變 (Sudden Spike / Drop)]
    if len(history_2min_windows) >= MIN_2MIN_BUFFER_COUNT:
        baseline_2min_avg = sum(history_2min_windows) / len(history_2min_windows)
        upper_bound = baseline_2min_avg * (1 + SPIKE_DROP_RATIO)
        lower_bound = baseline_2min_avg * (1 - SPIKE_DROP_RATIO)

        if window_bpm >= upper_bound or window_bpm <= lower_bound:
            status_code = "SUDDEN_HR_CHANGE"
            history_2min_windows.append(window_bpm)
            _save_anomaly_log(cursor, conn, current_time, window_bpm, status_code)
            return status_code

    # [第三層：持續性偏高]
    is_high = window_bpm > PERSISTENT_HIGH_BPM
    high_bpm_history.append(is_high)

    if len(high_bpm_history) == REQUIRED_HIGH_WINDOWS and all(high_bpm_history):
        status_code = "PERSISTENT_HIGH_HR"
    else:
        status_code = "NORMAL"

    history_2min_windows.append(window_bpm)
    _save_anomaly_log(cursor, conn, current_time, window_bpm, status_code)
    return status_code


def _save_anomaly_log(cursor, conn, timestamp, avg_bpm, status_code):
    """輔助函式：寫入分析紀錄至 anomaly_logs"""
    cursor.execute('''
        INSERT INTO anomaly_logs (timestamp, avg_bpm, status)
        VALUES (?, ?, ?)
    ''', (timestamp, round(avg_bpm, 1), status_code))
    conn.commit()
    print(f"\n[AI Evaluation] Time: {timestamp} | 15s Avg BPM: {avg_bpm:.1f} | Result: {status_code}\n")


def feed_bpm_stream(bpm, cursor, conn):
    """每秒採樣一次 BPM 累計進 15 秒視窗"""
    global last_window_sample_time
    now = time.time()

    # 排除無效心率，且每隔 1 秒取一次樣，確保視窗代表真實 15 秒
    if bpm > 0 and (now - last_window_sample_time >= 1.0):
        last_window_sample_time = now
        current_15s_window.append(bpm)

        if len(current_15s_window) >= WINDOW_SIZE_SEC:
            avg_bpm = sum(current_15s_window) / len(current_15s_window)
            current_15s_window.clear()
            evaluate_window_bpm(avg_bpm, cursor, conn)


# ==================== 3. 初始化資料庫 ====================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. 儲存高頻感測器資料表 (20Hz)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ax REAL, ay REAL, az REAL,
            gx REAL, gy REAL, gz REAL,
            ppg_red REAL, ppg_ir REAL,
            bpm REAL, spo2 REAL,
            gsr_smooth REAL, gsr_us REAL
        )
    ''')

    # 2. 儲存更新後的 PPG 異常診斷結果資料表 (每 15 秒一筆)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS anomaly_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            avg_bpm REAL,
            status TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print(f"[SQL] 資料庫 '{DB_NAME}' 初始化完成，具備感測資料表與異常狀態紀錄表！")


# ==================== 4. 主程序 ====================
def main():
    init_db()

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
        print(f"[Serial] 成功連接至 {COM_PORT}")
        print("[System] 開始接收 ESP32 生理與姿態數據並寫入 SQLite...\n")
    except Exception as e:
        print(f"[ERROR] 無法開啟 Port {COM_PORT}: {e}")
        return

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        while True:
            if ser.in_waiting > 0:
                raw_bytes = ser.readline()
                raw_str = raw_bytes.decode('utf-8', errors='ignore').strip()

                if not raw_str or raw_str.startswith("藍牙") or raw_str.startswith("rst:"):
                    continue

                parts = raw_str.split(',')

                # 判斷是否完整包含 12 個感測器欄位
                if len(parts) == 12:
                    try:
                        data_values = [float(p) for p in parts]
                        current_time = time.strftime('%Y-%m-%d %H:%M:%S')

                        # 1. 寫入原始感測器串流 (20Hz)
                        sql_query = '''
                            INSERT INTO sensor_data (
                                timestamp, ax, ay, az, gx, gy, gz, 
                                ppg_red, ppg_ir, bpm, spo2, gsr_smooth, gsr_us
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        '''
                        cursor.execute(sql_query, [current_time] + data_values)
                        conn.commit()

                        # 2. 取得即時 BPM 並餵入三層演算法評估 (資料庫同時記錄狀態)
                        bpm = data_values[8]
                        feed_bpm_stream(bpm, cursor, conn)

                        # 3. 終端機顯示狀態
                        print(f"[{current_time}] 寫入 -> AccX: {data_values[0]:<5.2f} | BPM: {data_values[8]:<5.1f} | SpO2: {data_values[9]:<5.1f}% | GSR: {data_values[10]:<6.1f}")

                    except ValueError:
                        pass

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[System] 手動關閉程式。")

    finally:
        ser.close()
        conn.close()
        print("[System] 連線已安全釋放。")


if __name__ == '__main__':
    main()