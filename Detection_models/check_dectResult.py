import sqlite3
import pandas as pd
from pathlib import Path

# 指向前一層資料夾的 detection_result.db
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = (BASE_DIR / ".." / "detection_result.db").resolve()

print("=" * 60)
print("🔍 檢查 detection_result.db 資料庫內容")
print(f"資料庫路徑: {DB_PATH}")
print("=" * 60)

if not DB_PATH.exists():
    print(f"❌ 找不到資料庫檔案: {DB_PATH}")
    print("請先執行 main_inference.py 產生推論結果。")
    exit()

conn = sqlite3.connect(str(DB_PATH))

# 1. 查詢總筆數
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM results")
total_count = cursor.fetchone()[0]
print(f"\n📊 目前累積推論結果筆數: {total_count} 筆\n")

if total_count > 0:
    # 2. 顯示最新 10 筆推論資料
    print("=== 最新 10 筆融合推論紀錄 ===")
    query = """
        SELECT id, timestamp, latest_sensor_id, imu_status, ppg_bpm, ppg_status, eda_value, eda_status, final_decision
        FROM results
        ORDER BY id DESC
        LIMIT 10
    """
    df = pd.read_sql_query(query, conn)
    # 調整 Pandas 顯示寬度
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(index=False))

    # 3. 統計最終決策狀態分佈
    print("\n" + "=" * 60)
    print("📈 最終決策狀態分佈 (final_decision):")
    stat_query = """
        SELECT final_decision, COUNT(*) as count 
        FROM results 
        GROUP BY final_decision 
        ORDER BY count DESC
    """
    df_stat = pd.read_sql_query(stat_query, conn)
    print(df_stat.to_string(index=False))

    # 4. 統計 IMU / PPG / EDA 個別狀態
    print("\n📈 IMU 狀態分佈:")
    print(pd.read_sql_query("SELECT imu_status, COUNT(*) as count FROM results GROUP BY imu_status", conn).to_string(index=False))
else:
    print("⚠️ 資料表內尚無任何資料。")

conn.close()
print("\n" + "=" * 60)