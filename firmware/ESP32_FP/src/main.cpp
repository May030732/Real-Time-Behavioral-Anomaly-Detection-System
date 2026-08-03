#include <Arduino.h>
#include <Wire.h>
#include "MPU6050_Module.h" // 引進模組
#include "Max30102_Module.h"

// put function declarations here:
int myFunction(int, int);

void setup()
{
  Serial.begin(115200);

  // 初始化 ESP32 的 I2C 引腳
  Wire.begin(21, 22);

  // 呼叫模組的初始化
  init_MPU6050();
  init_MAX30102();
}

void loop()
{
  MPUData imu = read_MPU6050();
  HeartRateData ppg = read_MAX30102();

  // 印出資料
  Serial.print("\n加速度原始值 => X: ");
  Serial.print(imu.ax);
  Serial.print(" | Y: ");
  Serial.print(imu.ay);
  Serial.print(" | Z: ");
  Serial.print(imu.az);

  Serial.print("  ||  陀螺儀原始值 => X: ");
  Serial.print(imu.gx);
  Serial.print(" | Y: ");
  Serial.print(imu.gy);
  Serial.print(" | Z: ");
  Serial.println(imu.gz);

  // MAX30102
  Serial.print("\nPPG_IR:");
  Serial.println(ppg.ir);

  delay(300);
}

// put function definitions here:
int myFunction(int x, int y)
{
  return x + y;
}