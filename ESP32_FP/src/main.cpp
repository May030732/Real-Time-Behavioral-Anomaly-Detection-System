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
  // 1. 【核心 100Hz 主採樣迴圈】確保時間週期精準（每 10ms 執行一次）
  MPUData imu = read_MPU6050();
  HeartRateData ppg = read_MAX30102();
  HealthReport result = process_PPG(ppg.red, ppg.ir);

  // 2. 當捕捉到心跳瞬間，立刻發送心跳報告 (同時輸出至 USB 串口與藍牙)
  if (result.isBeat)
  {
    current_bpm = result.bpm;
    if (result.spo2 > 0)
    {
      current_spo2 = result.spo2;
    }

    String beatMsg = "\n==================================================\n"
                     " 【❤️ 心跳觸發】 心率: " +
                     String(current_bpm, 1) +
                     " BPM  |  血氧 SpO2: " + String(current_spo2, 1) + " %\n"
                                                                        "==================================================";

    Serial.println(beatMsg);
    SerialBT.println(beatMsg); // 同步輸出至藍牙
  }

  // 3. 【GSR & 繪圖器資料】使用非阻塞計時器，控制在 50ms (20Hz) 輸出，保護採樣率
  static uint32_t last_gsr_time = 0;
  if (millis() - last_gsr_time >= 50)
  {
    last_gsr_time = millis();
    float gsrVal = gsr.readSmooth();
    float uS = gsr.readConductance();

    // 格式化輸出，利於 Arduino Serial Plotter 繪製二元曲線
    Serial.print("GSR_Smooth:");
    Serial.print(gsrVal);
    Serial.print(",");
    Serial.print("Conductance_uS:");
    Serial.println(uS);
  }

  // 4. 【除錯終端機】使用非阻塞計時器，每 500ms 輸出一次文字狀態
  static uint32_t last_print_time = 0;
  if (millis() - last_print_time >= 500)
  {
    last_print_time = millis();

    // 印出 MPU-6050 姿態
    Serial.print("【6050 IMU】 Acc => X:");
    Serial.print(imu.ax);
    Serial.print(" Y:");
    Serial.print(imu.ay);
    Serial.print(" Z:");
    Serial.print(imu.az);
    Serial.print("  |  Gyro => X:");
    Serial.print(imu.gx);
    Serial.print(" Y:");
    Serial.print(imu.gy);
    Serial.print(" Z:");
    Serial.println(imu.gz);

    // 印出 MAX30102 原始強度
    Serial.print("【30102 Raw】 PPG_IR: ");
    Serial.print(ppg.ir);
    Serial.print("  ||  PPG_RED: ");
    Serial.println(ppg.red);
    Serial.println("--------------------------------------------------");
  }

  // 5. 控制主迴圈延遲 10ms，達成穩定的 100Hz 採樣率
  delay(10);
}

// put function definitions here:
int myFunction(int x, int y)
{
  return x + y;
}