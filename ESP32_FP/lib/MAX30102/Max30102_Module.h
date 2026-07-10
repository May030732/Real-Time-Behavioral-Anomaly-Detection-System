#ifndef MAX30102_MODULE_H
#define MAX30102_MODULE_H

#include <Arduino.h>

// 1. 打包 MAX30102 輸出的生理數據結構
struct HeartRateData
{
    uint32_t red; // 紅光原始值（可用於計算心率）
    uint32_t ir;  // 紅外線原始值（可用於計算血氧 SpO2）
};

// 2. 宣告主程式可以呼叫的介面功能
void init_MAX30102();
bool available_MAX30102();
HeartRateData read_MAX30102();

#endif