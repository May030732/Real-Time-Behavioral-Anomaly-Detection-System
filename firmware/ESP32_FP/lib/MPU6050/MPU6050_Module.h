#ifndef MPU6050_MODULE_H
#define MPU6050_MODULE_H

#include <Arduino.h>

// 打包數據的結構定義
struct MPUData
{
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
};

// 宣告功能函式
void init_MPU6050();
MPUData read_MPU6050();

#endif