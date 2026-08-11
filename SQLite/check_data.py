import sqlite3
import pandas as pd

conn = sqlite3.connect("anomaly_detection.db")

# 用 pandas 一次將資料庫轉為表格印出
df = pd.read_sql_query("SELECT * FROM sensor_data", conn)

print(f"目前資料庫總共有 {len(df)} 筆資料：\n")
print(df.tail(100)) # 印出最後 10 筆

conn.close()