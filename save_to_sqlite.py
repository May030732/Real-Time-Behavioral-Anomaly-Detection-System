import serial
import sqlite3
import time
import os

# ==================== 1. 設定參數 ====================
# 資料庫檔案名稱 (會自動生成在與此 .py 相同的目錄下)
DB_NAME = "anomaly_detection.db"

# Serial 序列埠設定
# Windows 請確認 COM Port 號碼 (例如 'COM3', 'COM4')
# Mac / Linux 請改為例如 '/dev/tty.usbserial-xxx' 或 '/dev/ttyUSB0'
COM_PORT = 'COM3'  
BAUD_RATE = 115200

# ==================== 2. 初始化 SQLite 資料庫 ====================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 建立 sensor_data 資料表
    # 包含了 id, timestamp(時間), 以及多個感測器欄位 (可依需求增減)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            val1 REAL,
            val2 REAL,
            val3 REAL
        )
    ''')
    conn.commit()
    conn.close()
    print(f"[SQL] 資料庫 '{DB_NAME}' 初始化完成！")

# ==================== 3. 主程式：監聽 Serial 並寫入資料庫 ====================
def main():
    init_db()

    # 嘗試開啟 Serial 連線
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
        print(f"[Serial] 成功連接至 {COM_PORT} (Baud Rate: {BAUD_RATE})")
        print("[System] 開始接收 ESP32 資料並寫入 SQLite，按 Ctrl+C 可停止...\n")
    except Exception as e:
        print(f"[ERROR] 無法開啟 Port {COM_PORT}: {e}")
        print("請確認：1. ESP32 是否已插上電腦  2. COM Port 號碼是否正確  3. PlatformIO 的 Serial Monitor 是否已關閉")
        return

    # 連接資料庫準備寫入
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        while True:
            # 檢查是否有 Serial 資料傳入
            if ser.in_waiting > 0:
                # 讀取一行並去除兩端空格/換行符號
                raw_bytes = ser.readline()
                raw_str = raw_bytes.decode('utf-8', errors='ignore').strip()

                if not raw_str:
                    continue

                # 排除 ESP32 啟動時的系統除錯訊息（非純數字資料）
                if raw_str.startswith("rst:") or raw_str.startswith("ets"):
                    continue

                try:
                    # 假設 ESP32 傳來的格式為 "數值1,數值2,數值3" (例如 "0.12,9.81,-0.05")
                    data_parts = raw_str.split(',')

                    # 這裡以 3 個欄位為例，若 ESP32 傳 2 個或 4 個請自行修改長度判斷
                    if len(data_parts) == 3:
                        val1 = float(data_parts[0])
                        val2 = float(data_parts[1])
                        val3 = float(data_parts[2])

                        # 取得目前電腦精確時間 (包含年月日時分秒)
                        current_time = time.strftime('%Y-%m-%d %H:%M:%S')

                        # 執行 SQL 寫入指令
                        cursor.execute(
                            "INSERT INTO sensor_data (timestamp, val1, val2, val3) VALUES (?, ?, ?, ?)",
                            (current_time, val1, val2, val3)
                        )
                        conn.commit() # 確定儲存

                        print(f"[{current_time}] 存入成功 -> v1: {val1:<6.2f} | v2: {val2:<6.2f} | v3: {val3:<6.2f}")

                except ValueError:
                    # 抓取傳輸過程中封包破損或解包失敗的情況，避免程式報錯終止
                    print(f"[Warning] 收到無效封包 (已跳過): {raw_str}")

            time.sleep(0.01) # 短暫休息避免 CPU 佔用率過高

    except KeyboardInterrupt:
        print("\n[System] 使用者手動停止程式。")

    finally:
        # 安全關閉資源
        ser.close()
        conn.close()
        print("[System] Serial 與 資料庫連線已安全關閉。")

if __name__ == '__main__':
    main()