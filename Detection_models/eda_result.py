import sqlite3
import time
import os

# ==========================================
# EDA Arousal Detection
# EDA -> Baseline -> Arousal Detection
# ==========================================

# 建立 baseline 要收集幾筆 EDA
BASELINE_SAMPLE_COUNT = 30

# EDA 相對 baseline 上升 20%
# 進入 Suspected_Arousal 範圍
SUSPECTED_THRESHOLD = 0.20

# EDA 相對 baseline 上升 50%
# 進入 High_Arousal 範圍
HIGH_THRESHOLD = 0.50

# 必須連續幾次超過門檻才正式判定
REQUIRED_ABNORMAL_COUNT = 3

# 指向最外層 SQLite 資料庫
DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "anomaly_detection.db"
)


# ==========================================
# 變數初始化
# ==========================================

# 用來建立 baseline
baseline_values = []

# 個人 EDA baseline
baseline_eda = None

# Suspected_Arousal 連續次數
suspected_count = 0

# High_Arousal 連續次數
high_count = 0

# 記錄上一筆處理過的資料 id
# 避免重複判斷同一筆資料
last_processed_id = None


# ==========================================
# EDA 異常判斷函式
# ==========================================

def detect_eda(current_eda):

    global baseline_eda
    global suspected_count
    global high_count

    # --------------------------------------
    # 1. 排除無效 EDA
    # --------------------------------------

    if current_eda is None:
        print("EDA: None | Invalid EDA (Skipped)")
        return "Invalid_EDA"

    if current_eda <= 0:
        print(
            f"EDA: {current_eda:.3f}"
            f" | Invalid EDA (Skipped)"
        )
        return "Invalid_EDA"


    # --------------------------------------
    # 2. 建立個人 Baseline
    # --------------------------------------

    if baseline_eda is None:

        baseline_values.append(current_eda)

        print(
            f"Collecting baseline: "
            f"{len(baseline_values)}/{BASELINE_SAMPLE_COUNT}"
            f" | EDA: {current_eda:.3f}"
        )

        # 收集完成
        if len(baseline_values) >= BASELINE_SAMPLE_COUNT:

            baseline_eda = (
                sum(baseline_values)
                / len(baseline_values)
            )

            print("\n============================")
            print(
                f"EDA Baseline: "
                f"{baseline_eda:.3f}"
            )
            print("EDA Baseline completed")
            print("============================\n")

        return "Collecting_Baseline"


    # --------------------------------------
    # 3. 計算 EDA 與 Baseline 的差異
    # --------------------------------------

    eda_change = current_eda - baseline_eda

    # 算相對變化比例
    #
    # 例如：
    # baseline = 4
    # current = 5
    #
    # (5 - 4) / 4 = 0.25
    #
    # 代表上升 25%
    change_ratio = eda_change / baseline_eda


    # --------------------------------------
    # 4. High Arousal
    # --------------------------------------

    if change_ratio >= HIGH_THRESHOLD:

        high_count += 1

        # High 時不累積 suspected
        suspected_count = 0

        # 必須連續出現指定次數
        if high_count >= REQUIRED_ABNORMAL_COUNT:

            status = "High_Arousal"

        else:

            # 還沒連續滿 3 次
            status = "Normal"


    # --------------------------------------
    # 5. Suspected Arousal
    # --------------------------------------

    elif change_ratio >= SUSPECTED_THRESHOLD:

        suspected_count += 1

        # 目前不是 High
        high_count = 0

        # 必須連續出現指定次數
        if suspected_count >= REQUIRED_ABNORMAL_COUNT:

            status = "Suspected_Arousal"

        else:

            # 還沒滿 3 次
            status = "Normal"


    # --------------------------------------
    # 6. Normal
    # --------------------------------------

    else:

        suspected_count = 0
        high_count = 0

        status = "Normal"


    # --------------------------------------
    # 7. 顯示判斷結果
    # --------------------------------------

    print(
        f"EDA: {current_eda:.3f}"
        f" | Baseline: {baseline_eda:.3f}"
        f" | Change: {eda_change:+.3f}"
        f" | Ratio: {change_ratio * 100:+.1f}%"
        f" | Suspected Count: {suspected_count}"
        f" | High Count: {high_count}"
        f" | Status: {status}"
    )


    return status


# ==========================================
# 從 SQLite 讀取最新 EDA 資料
# ==========================================

def get_latest_eda():

    global last_processed_id


    # --------------------------------------
    # 檢查資料庫是否存在
    # --------------------------------------

    if not os.path.exists(DB_PATH):

        print(
            f"[Warning] 找不到資料庫檔案: "
            f"{DB_PATH}"
        )

        return None


    try:

        # timeout=5
        # 避免另一支程式正在寫 SQLite 時
        # 馬上產生 database is locked
        conn = sqlite3.connect(
            DB_PATH,
            timeout=5
        )

        cursor = conn.cursor()


        # --------------------------------------
        # 讀取最新一筆資料
        #
        # 這裡使用 gsr_smooth
        # --------------------------------------

        cursor.execute(
            """
            SELECT id, gsr_smooth
            FROM sensor_data
            ORDER BY id DESC
            LIMIT 1
            """
        )


        row = cursor.fetchone()

        conn.close()


        # --------------------------------------
        # 有讀到資料
        # --------------------------------------

        if row:

            latest_id = row[0]

            latest_eda = row[1]


            # ----------------------------------
            # 防止重複處理同一筆資料
            # ----------------------------------

            if latest_id == last_processed_id:

                return None


            # 更新最後處理的 id
            last_processed_id = latest_id


            return latest_eda


    # ======================================
    # SQLite 正在被另一個程式寫入
    # ======================================

    except sqlite3.OperationalError as e:

        print(
            f"[SQL Busy] "
            f"讀取時資料庫正在寫入: {e}"
        )

        return None


    # ======================================
    # 其他 SQLite 錯誤
    # ======================================

    except sqlite3.Error as e:

        print(
            f"[SQL Error] "
            f"資料庫發生錯誤: {e}"
        )

        return None


    return None


# ==========================================
# 主程式
# 每隔 1 秒讀取最新 EDA
# ==========================================

def main():

    print("=" * 60)

    print(
        "開始監聽 anomaly_detection.db "
        "的最新 EDA 數據..."
    )

    print(
        f"資料庫目標路徑: "
        f"{os.path.abspath(DB_PATH)}"
    )

    print(f"Baseline 樣本數: {BASELINE_SAMPLE_COUNT}")

    print(
        f"Suspected threshold: "
        f"+{SUSPECTED_THRESHOLD * 100:.0f}%"
    )

    print(
        f"High threshold: "
        f"+{HIGH_THRESHOLD * 100:.0f}%"
    )

    print(
        f"Required consecutive count: "
        f"{REQUIRED_ABNORMAL_COUNT}"
    )

    print("=" * 60 + "\n")


    try:

        while True:

            # ----------------------------------
            # 讀取最新一筆 EDA
            # ----------------------------------

            latest_eda = get_latest_eda()


            # ----------------------------------
            # 有新資料才進行判斷
            # ----------------------------------

            if latest_eda is not None:

                detect_eda(latest_eda)


            # 每隔 1 秒檢查一次資料庫
            time.sleep(1)


    except KeyboardInterrupt:

        print(
            "\n[System] "
            "已手動停止 EDA Arousal 檢測程式。"
        )


# ==========================================
# 程式入口
# ==========================================

if __name__ == '__main__':

    main()