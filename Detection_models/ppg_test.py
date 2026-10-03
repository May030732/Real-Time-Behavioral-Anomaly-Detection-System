import sqlite3
import time
import os
from collections import deque

# ==========================================
# 參數與門檻設定
# ==========================================

# 時間視窗設定 (單位：秒/筆數)
WINDOW_SIZE_SEC = 15          # 每 15 秒算一次平均 BPM

# 第一層：極限值門檻
EXTREME_LOW_BPM = 40
EXTREME_HIGH_BPM = 180
REQUIRED_EXTREME_WINDOWS = 2  # 連續 2 個視窗 (30秒)

# 第二層：突發性劇變門檻
SPIKE_DROP_RATIO = 0.4        # 比前 2 分鐘平均值高/低 40%
MIN_2MIN_BUFFER_COUNT = 4     # 至少需要 4 個視窗 (1分鐘) 的歷史資料才進行突發判斷
MAX_2MIN_BUFFER_COUNT = 8     # 最多保留 8 個視窗 (2分鐘) 的歷史資料

# 第三層：持續性偏高門檻
PERSISTENT_HIGH_BPM = 100
REQUIRED_HIGH_WINDOWS = 4     # 連續 4 個視窗 (1分鐘)

# 指向 SQLite 資料庫檔
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "anomaly_detection.db")


# ==========================================
# 狀態佇列與變數初始化
# ==========================================

current_15s_window = []       # 用於收集當前 15 秒內的 BPM 樣本
history_2min_windows = deque(maxlen=MAX_2MIN_BUFFER_COUNT) # 儲存過去 2 分鐘的視窗平均值

# 狀態歷史紀錄（用於判斷「連續成立」條件）
extreme_status_history = deque(maxlen=REQUIRED_EXTREME_WINDOWS) # 記錄極端值的連續狀態
high_bpm_history = deque(maxlen=REQUIRED_HIGH_WINDOWS)         # 記錄偏高的連續狀態

last_processed_id = None      # 避免重複讀取相同數據


# ==========================================
# 三層異常判斷核心 logic
# ==========================================

def evaluate_window_bpm(window_bpm):
    """
    輸入 15 秒視窗的平均 BPM，依據三層邏輯進行判定並回傳簡短英文描述
    """
    print(f"\n[Window Complete] 15s Avg BPM: {window_bpm:.1f}")

    # -------------------------------------------------------------
    # [ 第一層：極限值判斷 ]
    # -------------------------------------------------------------
    is_extreme = (window_bpm < EXTREME_LOW_BPM) or (window_bpm > EXTREME_HIGH_BPM)
    extreme_status_history.append(is_extreme)

    # 檢查是否連續 2 個視窗 (30秒) 皆成立
    if len(extreme_status_history) == REQUIRED_EXTREME_WINDOWS and all(extreme_status_history):
        status_code = "CRITICAL_EXTREME_HR"
        print(f"--> Triggered L1: {status_code} (BPM < 40 or > 180 for 30s)")
        
        history_2min_windows.append(window_bpm)
        return status_code

    # -------------------------------------------------------------
    # [ 第二層：突發性劇變 (Sudden Spike / Drop) ]
    # -------------------------------------------------------------
    if len(history_2min_windows) >= MIN_2MIN_BUFFER_COUNT:
        baseline_2min_avg = sum(history_2min_windows) / len(history_2min_windows)
        upper_bound = baseline_2min_avg * (1 + SPIKE_DROP_RATIO)  # 高出 40%
        lower_bound = baseline_2min_avg * (1 - SPIKE_DROP_RATIO)  # 低於 40%

        if window_bpm >= upper_bound or window_bpm <= lower_bound:
            status_code = "SUDDEN_HR_CHANGE"
            print(f"--> Triggered L2: {status_code} (Current: {window_bpm:.1f} | 2min Avg: {baseline_2min_avg:.1f})")
            
            history_2min_windows.append(window_bpm)
            return status_code

    # -------------------------------------------------------------
    # [ 第三層：持續性偏高 ]
    # -------------------------------------------------------------
    is_high = window_bpm > PERSISTENT_HIGH_BPM
    high_bpm_history.append(is_high)

    # 檢查是否連續 4 個視窗 (1分鐘) BPM > 100
    if len(high_bpm_history) == REQUIRED_HIGH_WINDOWS and all(high_bpm_history):
        status_code = "PERSISTENT_HIGH_HR"
        print(f"--> Triggered L3: {status_code} (BPM > 100 for 1min)")
    else:
        status_code = "NORMAL"
        print(f"--> Status: {status_code}")

    history_2min_windows.append(window_bpm)
    return status_code


# ==========================================
# 收集 15 秒數據並驅動檢測
# ==========================================

def process_bpm_stream(current_bpm):
    if current_bpm <= 0:
        return None

    current_15s_window.append(current_bpm)
    print(f"Collecting 15s window... ({len(current_15s_window)}/{WINDOW_SIZE_SEC}) | Current BPM: {current_bpm:.1f}")

    if len(current_15s_window) >= WINDOW_SIZE_SEC:
        window_avg_bpm = sum(current_15s_window) / len(current_15s_window)
        current_15s_window.clear()
        
        return evaluate_window_bpm(window_avg_bpm)

    return None


# ==========================================
# 從 SQLite 讀取最新 BPM 資料
# ==========================================

def get_latest_bpm():
    global last_processed_id
    
    if not os.path.exists(DB_PATH):
        print(f"[Warning] Database file not found: {DB_PATH}")
        return None

    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        cursor = conn.cursor()

        cursor.execute("SELECT id, bpm FROM sensor_data ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        if row:
            latest_id, latest_bpm = row[0], row[1]
            
            if latest_id == last_processed_id:
                return None
            
            last_processed_id = latest_id
            return latest_bpm

    except sqlite3.OperationalError as e:
        print(f"[SQL Busy] Database locked: {e}")
        return None

    return None


# ==========================================
# 主迴圈
# ==========================================

def main():
    print("=" * 60)
    print("Starting 3-Layer PPG Anomaly Detection System (15s Window)...")
    print(f"Database Target: {os.path.abspath(DB_PATH)}")
    print("=" * 60 + "\n")

    try:
        while True:
            latest_bpm = get_latest_bpm()
            
            if latest_bpm is not None:
                process_bpm_stream(latest_bpm)
            
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[System] PPG Anomaly Detection stopped manually.")

if __name__ == '__main__':
    main()