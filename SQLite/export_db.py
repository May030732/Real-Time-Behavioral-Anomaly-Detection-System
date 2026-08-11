import sqlite3
import pandas as pd
import os

DB_NAME = "anomaly_detection.db"
# 你可以選擇要匯出原始資料表 (sensor_data) 或是 特徵資料表 (feature_summary)
TABLE_NAME = "feature_summary" 

def export_data():
    if not os.path.exists(DB_NAME):
        print(f"[錯誤] 找不到資料庫檔案 '{DB_NAME}'！")
        return

    # 1. 連接資料庫
    conn = sqlite3.connect(DB_NAME)

    # 2. 讀取 SQLite 表格並轉成 Pandas DataFrame
    # 這裡可以用 SQL 語法進行排序或初步篩選
    query = f"SELECT * FROM {TABLE_NAME} ORDER BY id ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print(f"[提示] 資料表 '{TABLE_NAME}' 中沒有任何數據。")
        return

    print(f"成功從 SQLite 讀取 {len(df)} 筆資料！")

    # 3. 匯出為 Excel 檔 (.xlsx) - 人類最方便編輯的格式
    excel_file = f"{TABLE_NAME}_dataset.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"已成功匯出至 Excel 檔：{excel_file}")

    # 4. 同時匯出為 CSV 檔 (.csv) - 機器學習模型訓練常用格式
    csv_file = f"{TABLE_NAME}_dataset.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig') # utf-8-sig 確保 Excel 開啟中文不亂碼
    print(f"已成功匯出至 CSV 檔：{csv_file}")

if __name__ == "__main__":
    export_data()