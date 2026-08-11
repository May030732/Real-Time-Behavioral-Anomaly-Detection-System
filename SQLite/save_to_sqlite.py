import serial
import sqlite3
import time

# ==================== 1. 設定參數 ====================
DB_NAME = "anomaly_detection.db"

# Serial Port 設定 (若改用藍牙，這裡指定藍牙配對後的 COM Port 即可)
COM_PORT = 'COM4'  # 請修改為您電腦對應的 COM Port 號碼
BAUD_RATE = 115200

# ==================== 2. 初始化資料庫 ====================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 建立多感測器欄位的資料表
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
    conn.commit()
    conn.close()
    print(f"[SQL] 資料庫 '{DB_NAME}' 初始化完成，具備多模組生理與姿態欄位！")

# ==================== 3. 主程序 ====================
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

                # 依據逗號拆解資料
                parts = raw_str.split(',')

                # 判斷是否完整包含 12 個感測器欄位
                if len(parts) == 12:
                    try:
                        # 解析成浮點數
                        data_values = [float(p) for p in parts]
                        current_time = time.strftime('%Y-%m-%d %H:%M:%S')

                        # 插入 12 個數據值到資料庫
                        sql_query = '''
                            INSERT INTO sensor_data (
                                timestamp, ax, ay, az, gx, gy, gz, 
                                ppg_red, ppg_ir, bpm, spo2, gsr_smooth, gsr_us
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        '''
                        cursor.execute(sql_query, [current_time] + data_values)
                        conn.commit()

                        # 即時印出關鍵數據供確認 (以 Acc, BPM, GSR 為例)
                        print(f"[{current_time}] 寫入成功 -> AccX: {data_values[0]:<5.2f} | BPM: {data_values[8]:<5.1f} | GSR: {data_values[10]:<6.1f}")

                    except ValueError:
                        # 忽略單次雜訊或解包失敗
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