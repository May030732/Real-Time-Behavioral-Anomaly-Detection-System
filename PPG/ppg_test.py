import sqlite3
import time
import os

# ==========================================
# PPG BPM Arousal Detection
# BPM -> Baseline -> Abnormal Detection
# ==========================================

# 建立 baseline 要收集幾筆 BPM
BASELINE_SAMPLE_COUNT = 30

# 比 baseline 增加多少 BPM，先視為疑似 arousal
BPM_CHANGE_THRESHOLD = 25

# 必須連續幾次超過門檻
REQUIRED_ABNORMAL_COUNT = 3

# 指向最外層的 SQLite 資料庫檔
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "anomaly_detection.db")


# ==========================================
# 變數初始化
# ==========================================

baseline_values = []
baseline_bpm = None
abnormal_count = 0
last_processed_id = None  # 用於紀錄上一筆處理過的資料 id，避免重複讀取相同數據


# ==========================================
# 異常判斷函式
# ==========================================

def detect_arousal(current_bpm):
    global baseline_bpm
    global abnormal_count

    # 1. 排除無效 BPM
    if current_bpm <= 0:
        print(f"BPM: {current_bpm:.1f} | Invalid BPM (Skipped)")
        return "Invalid_BPM"

    # 2. 建立 Baseline
    if baseline_bpm is None:
        baseline_values.append(current_bpm)
        print(
            f"Collecting baseline: "
            f"{len(baseline_values)}/{BASELINE_SAMPLE_COUNT} "
            f"| BPM: {current_bpm:.1f}"
        )

        # 收集完成
        if len(baseline_values) >= BASELINE_SAMPLE_COUNT:
            baseline_bpm = sum(baseline_values) / len(baseline_values)
            print("\n============================")
            print(f"Baseline BPM: {baseline_bpm:.2f}")
            print("Baseline completed")
            print("============================\n")

        return "Collecting_Baseline"

    # 3. 計算 BPM Change
    bpm_change = current_bpm - baseline_bpm

    # 4. 判斷是否超過門檻
    if bpm_change >= BPM_CHANGE_THRESHOLD:
        abnormal_count += 1
    else:
        abnormal_count = 0

    # 5. 最終判斷
    if abnormal_count >= REQUIRED_ABNORMAL_COUNT:
        status = "Suspected_Arousal"
    else:
        status = "Normal"

    # 6. 顯示結果
    print(
        f"BPM: {current_bpm:.1f}"
        f" | Baseline: {baseline_bpm:.1f}"
        f" | Change: {bpm_change:+.1f}"
        f" | Count: {abnormal_count}"
        f" | Status: {status}"
    )

    return status


# ==========================================
# 從 SQLite 讀取最新 BPM 資料
# ==========================================

def get_latest_bpm():
    global last_processed_id
    
    if not os.path.exists(DB_PATH):
        print(f"[Warning] 找不到資料庫檔案: {DB_PATH}")
        return None

    try:
        # timeout=5 防止 save_to_sqlite 寫入時造成的短暫資料庫鎖定
        conn = sqlite3.connect(DB_PATH, timeout=5)
        cursor = conn.cursor()

        # 撈出最新的一筆 id 與 bpm
        cursor.execute("SELECT id, bpm FROM sensor_data ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        if row:
            latest_id, latest_bpm = row[0], row[1]
            
            # 若與上一筆 id 相同，代表目前沒有新數據傳入
            if latest_id == last_processed_id:
                return None
            
            last_processed_id = latest_id
            return latest_bpm

    except sqlite3.OperationalError as e:
        print(f"[SQL Busy] 讀取時資料庫正忙著寫入中: {e}")
        return None

    return None


# ==========================================
# 主迴圈：每隔 1 秒自動檢查與判斷
# ==========================================

def main():
    print("=" * 50)
    print("開始監聽 anomaly_detection.db 的最新 BPM 數據...")
    print(f"資料庫目標路徑: {os.path.abspath(DB_PATH)}")
    print("=" * 50 + "\n")

    try:
        while True:
            latest_bpm = get_latest_bpm()
            
            if latest_bpm is not None:
                detect_arousal(latest_bpm)
            
            # 每隔 1 秒讀取並判斷一次
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[System] 已手動停止 PPG Arousal 檢測程式。")

if __name__ == '__main__':
    main()