#include <Arduino.h>
#include <Wire.h>
#include "MPU6050_Module.h" // include lib
#include "Max30102_Module.h"
#include "Bio_Algorithm.h"
#include "BluetoothSerial.h"
BluetoothSerial SerialBT;

float current_bpm = 0.0;
float current_spo2 = 0.0;

// put function declarations here:
int myFunction(int, int);

void setup()
{
  Serial.begin(115200);
  SerialBT.begin("ESP32_Blanc");
  Serial.println("藍牙已啟動");
  // 初始化 ESP32 的 I2C 引腳
  Wire.begin(21, 22);

  // 叫模組初始化
  init_MPU6050();
  init_MAX30102();
}

void loop()
{
  // 1. 高速讀取感測器與執行演算法 (維持 100Hz 採樣率)
  MPUData imu = read_MPU6050();
  HeartRateData ppg = read_MAX30102();
  HealthReport result = process_PPG(ppg.red, ppg.ir);

  // 2. 當捕捉到心跳瞬間，立刻強行插播生理數據報告
  if (result.isBeat)
  {
    current_bpm = result.bpm;
    if (result.spo2 > 0)
    {
      current_spo2 = result.spo2;
    }

    Serial.println("\n==================================================================");
    Serial.print("  【❤️ 心跳觸發】 心率: ");
    Serial.print(current_bpm, 1);
    Serial.print(" BPM  |  血氧 SpO2: ");
    Serial.print(current_spo2, 1);
    Serial.println(" %");
    Serial.println("==================================================================");
  }

  // 3. 使用非阻塞計時器，每 500 毫秒才印一次感測器原始狀態（避免終端機刷太快當掉）
  static uint32_t last_print_time = 0;
  if (millis() - last_print_time > 500)
  {
    last_print_time = millis();

    // 印出 MPU-6050 資料
    Serial.print("【6050 IMU】 ");
    Serial.print("Acc => X:");
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
    Serial.print(imu.gz);
    Serial.println();

    // 印出 MAX30102 原始反射光強度
    Serial.print("【30102 Raw】");
    Serial.print("PPG_IR: ");
    Serial.print(ppg.ir);
    Serial.print("  ||  PPG_RED: ");
    Serial.print(ppg.red);
    Serial.println();

    Serial.println("------------------------------------------------------------------");
  }

  // 4. 【最核心！】將延遲縮短到 10 毫秒，維持高精度採樣
  delay(10);
}

// put function definitions here:
int myFunction(int x, int y)
{
  return x + y;
}