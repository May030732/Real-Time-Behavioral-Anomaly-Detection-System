#include <Arduino.h>
#include <Wire.h>
#include "MPU6050_Module.h"
#include "Max30102_Module.h"
#include "Bio_Algorithm.h"
#include "BluetoothSerial.h"
#include "GSR_Sensor.h"

BluetoothSerial SerialBT;
GSR_Sensor gsr(34, 0.15);

float current_bpm = 0.0;
float current_spo2 = 0.0;

void setup()
{
  Serial.begin(115200);
  SerialBT.begin("ESP32_Blanc");
  Serial.println("藍牙與串口初始化完成...");

  // 初始化 ESP32 的 I2C 引腳 (SDA:21, SCL:22)
  Wire.begin(21, 22);

  // 初始化各模組
  init_MPU6050();
  init_MAX30102();
  gsr.begin();
}

void loop()
{
  // 1. 【核心 100Hz 主採樣迴圈】
  MPUData imu = read_MPU6050();
  HeartRateData ppg = read_MAX30102();
  HealthReport result = process_PPG(ppg.red, ppg.ir);

  // 2. 當捕捉到心跳瞬間，更新數值
  if (result.isBeat)
  {
    current_bpm = result.bpm;
    if (result.spo2 > 0)
    {
      current_spo2 = result.spo2;
    }
  }

  // 3. 【數據發送區塊】非阻塞計時器，每 50ms (20Hz) 發送一次 CSV 格式給 Python
  static uint32_t last_send_time = 0;
  if (millis() - last_send_time >= 50)
  {
    last_send_time = millis();

    float gsrVal = gsr.readSmooth();
    float uS = gsr.readConductance();

    /*
      定義輸出數據格式 (CSV 格式，用逗號分隔)：
      欄位顺序: ax, ay, az, gx, gy, gz, ppg_red, ppg_ir, bpm, spo2, gsr_smooth, gsr_us
    */
    String dataStr = String(imu.ax) + "," +
                     String(imu.ay) + "," +
                     String(imu.az) + "," +
                     String(imu.gx) + "," +
                     String(imu.gy) + "," +
                     String(imu.gz) + "," +
                     String(ppg.red) + "," +
                     String(ppg.ir) + "," +
                     String(current_bpm, 1) + "," +
                     String(current_spo2, 1) + "," +
                     String(gsrVal, 2) + "," +
                     String(uS, 2);

    // 同步透過 USB Serial 與藍牙發送純數據
    Serial.println(dataStr);
    SerialBT.println(dataStr);
  }

  // 4. 控制主迴圈延遲 10ms，達成穩定的 100Hz 採樣率
  delay(10);
}