import serial

# 請修改為你實際的 COM Port (例如 'COM3' 或 'COM4')
COM_PORT = 'COM4' 
BAUD_RATE = 115200

try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
    print(f"成功開啟 {COM_PORT}，等待 ESP32 資料...")
    
    # 印出前 10 筆接收到的 raw 資料
    for i in range(10):
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"[{i+1}] 收到: {line}")
            
    ser.close()
    print("Serial 讀取測試完成！")
except Exception as e:
    print(f"連線失敗: {e}")