#include "MPU6050_Module.h"
#include <Wire.h>

const int MPU_addr = 0x68;

void init_MPU6050()
{
    // 喚醒 MPU-6050
    Wire.beginTransmission(MPU_addr);
    Wire.write(0x6B);
    Wire.write(0);
    byte error = Wire.endTransmission();

    if (error == 0)
    {
        Serial.println("[Module] MPU-6050 喚醒成功！");
    }
    else
    {
        Serial.println("[Module] 【錯誤】無法喚醒 MPU-6050，請檢查接線。");
    }
}

MPUData read_MPU6050()
{
    MPUData data;

    Wire.beginTransmission(MPU_addr);
    Wire.write(0x3B);
    Wire.endTransmission(false);

    Wire.requestFrom(MPU_addr, 14, true);

    data.ax = Wire.read() << 8 | Wire.read();
    data.ay = Wire.read() << 8 | Wire.read();
    data.az = Wire.read() << 8 | Wire.read();

    // 跳過溫度暫存器的 2 位元組
    Wire.read();
    Wire.read();

    data.gx = Wire.read() << 8 | Wire.read();
    data.gy = Wire.read() << 8 | Wire.read();
    data.gz = Wire.read() << 8 | Wire.read();

    return data;
}