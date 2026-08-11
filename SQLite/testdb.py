import sqlite3
import time

# 測試建立資料庫
conn = sqlite3.connect("anomaly_detection.db")
cursor = conn.cursor()

# 建立表格
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

# 模擬一筆測試數據 (12 個數值)
test_data = [0.01, 0.02, 0.98, 0.1, 0.2, 0.3, 150000, 160000, 75.0, 98.0, 300.5, 3.3]
current_time = time.strftime('%Y-%m-%d %H:%M:%S')

sql = '''
    INSERT INTO sensor_data (
        timestamp, ax, ay, az, gx, gy, gz, 
        ppg_red, ppg_ir, bpm, spo2, gsr_smooth, gsr_us
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
'''

cursor.execute(sql, [current_time] + test_data)
conn.commit()

# 讀取剛剛寫入的資料驗證
cursor.execute("SELECT * FROM sensor_data ORDER BY id DESC LIMIT 1")
row = cursor.fetchone()

print(" SQL 測試成功！剛剛寫入的資料為：")
print(row)

conn.close()    