# 實時姿態與生理數據 SQLite 資料庫規格說明書 (DATABASE_DOC.md)

本專案採用 **SQLite** 作為輕量化、檔案型的嵌入式資料庫，用於保存由 ESP32 微控制器透過 Serial / 藍牙傳輸的 12 項原始感測器數據（MPU-6050 姿態、MAX30102 生理訊號與 GSR 皮膚電導數據）。

---

## 一、 目前 SQLite 擷取的資料欄位 specification

資料庫檔案名稱：`anomaly_detection.db`  
主要資料表名稱：`sensor_data`

| 欄位名稱 (Column) | 資料型態 (Type) | 說明 (Description) | 單位/範圍 | 來源模組 |
| :--- | :--- | :--- | :--- | :--- |
| **`id`** | `INTEGER` | 主鍵 (Primary Key)，自動遞增編號 | 1, 2, 3... | SQLite 底層生成 |
| **`timestamp`** | `DATETIME` | 資料寫入時的電腦系統時間戳記 | `YYYY-MM-DD HH:MM:SS` | 電腦端 Python 產生 |
| **`ax`** | `REAL` | MPU-6050 加速度計 X 軸 | g ($m/s^2$) | MPU-6050 |
| **`ay`** | `REAL` | MPU-6050 加速度計 Y 軸 | g ($m/s^2$) | MPU-6050 |
| **`az`** | `REAL` | MPU-6050 加速度計 Z 軸 | g ($m/s^2$) | MPU-6050 |
| **`gx`** | `REAL` | MPU-6050 陀螺儀 X 軸角速度 | °/s | MPU-6050 |
| **`gy`** | `REAL` | MPU-6050 陀螺儀 Y 軸角速度 | °/s | MPU-6050 |
| **`gz`** | `REAL` | MPU-6050 陀螺儀 Z 軸角速度 | °/s | MPU-6050 |
| **`ppg_red`** | `REAL` | MAX30102 紅光通道原始強度 | Raw ADC | MAX30102 |
| **`ppg_ir`** | `REAL` | MAX30102 紅外光通道原始強度 | Raw ADC | MAX30102 |
| **`bpm`** | `REAL` | 即時演算心率數值 | BPM (次/分) | Bio_Algorithm |
| **`spo2`** | `REAL` | 即時演算血氧濃度 | % | Bio_Algorithm |
| **`gsr_smooth`** | `REAL` | 平滑化後的皮膚電阻/電壓數值 | Raw ADC / Volt | GSR Sensor |
| **`gsr_us`** | `REAL` | 皮膚電導率 (Conductance) | $\mu S$ (微西門子) | GSR Sensor |

---

## 二、 系統數據流與程式架構圖

數據採集與資料庫持久化的整體運作架構如下：

```text
[ ESP32 採集端 ]
  ├── MPU-6050 (6-axis IMU)
  ├── MAX30102 (PPG / HeartRate)
  └── GSR Sensor (Conductance)
       │
       ▼ (100Hz 採樣，以 20Hz 頻率打包為 CSV 字串)
[ Serial (USB) / Bluetooth (COM4) ]
       │
       ▼ (以逗號切分 12 個感測器數據)
[ 電腦端 Python 監聽程式: save_to_sqlite.py ]
       │
       ▼ (封包自動校驗與時間戳記注入)
[ SQLite 資料庫檔案: anomaly_detection.db ]
       │
       ├── (離線特徵工程 / 滑動視窗計算)
       ├── (匯出成 Excel / CSV 供標記)
       └── (機器學習模型訓練輸入: SVM / Isolation Forest)

```

---

## 三、 核心程式碼與用途說明

專案中包含 3 個關鍵 Python 腳本，分別用於數據寫入、讀取檢視與特徵工程加工：

### 1. 寫入資料庫：`save_to_sqlite.py`

* **用途：** 負責連線 COM Port，監聽 ESP32 傳入的 CSV 數據字串，進行例外處理（跳過無效封包）並即時寫入 SQLite。

```python
import serial
import sqlite3
import time

DB_NAME = "anomaly_detection.db"
COM_PORT = 'COM4'
BAUD_RATE = 115200

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ax REAL, ay REAL, az REAL, gx REAL, gy REAL, gz REAL,
            ppg_red REAL, ppg_ir REAL, bpm REAL, spo2 REAL,
            gsr_smooth REAL, gsr_us REAL
        )
    ''')
    conn.commit()
    conn.close()

def main():
    init_db()
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        while True:
            if ser.in_waiting > 0:
                raw_str = ser.readline().decode('utf-8', errors='ignore').strip()
                parts = raw_str.split(',')

                if len(parts) == 12:
                    try:
                        data_values = [float(p) for p in parts]
                        current_time = time.strftime('%Y-%m-%d %H:%M:%S')

                        sql = '''
                            INSERT INTO sensor_data (
                                timestamp, ax, ay, az, gx, gy, gz, 
                                ppg_red, ppg_ir, bpm, spo2, gsr_smooth, gsr_us
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        '''
                        cursor.execute(sql, [current_time] + data_values)
                        conn.commit()
                        print(f"[{current_time}] 寫入成功")
                    except ValueError:
                        pass
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        conn.close()

if __name__ == '__main__':
    main()

```

---

### 2. 讀取與驗證資料：`check_db.py`

* **用途：** 調用 Pandas 載入 SQLite 資料表，快速確認資料庫總筆數與最後數筆資料狀態。

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("anomaly_detection.db")

# 用 Pandas 載入最新 10 筆資料
df = pd.read_sql_query("SELECT * FROM sensor_data ORDER BY id DESC LIMIT 10", conn)
print("=== 最新 10 筆感測器紀錄 ===")
print(df)

cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM sensor_data")
print(f"\n目前累積資料筆數：{cursor.fetchone()[0]} 筆")

conn.close()

```

---

### 3. 離線轉存特徵與 Excel 檔：`export_dataset.py`

* **用途：** 將原始資料庫讀出並轉存為 CSV 與 Excel 檔，方便團隊成員手動標記（Labeling）及準備 AI 模型訓練。

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("anomaly_detection.db")
df = pd.read_sql_query("SELECT * FROM sensor_data", conn)
conn.close()

# 匯出至 Excel 與 CSV
df.to_excel("sensor_dataset.xlsx", index=False)
df.to_csv("sensor_dataset.csv", index=False, encoding='utf-8-sig')
print(f"成功匯出 {len(df)} 筆資料至 sensor_dataset.xlsx 與 sensor_dataset.csv！")

```

---