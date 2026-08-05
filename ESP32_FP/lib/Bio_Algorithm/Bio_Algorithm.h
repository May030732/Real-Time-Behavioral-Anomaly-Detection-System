#ifndef BIO_ALGORITHM_H
#define BIO_ALGORITHM_H

#include <Arduino.h>

// 定義一個結構體，用來打包最終的生理報告
struct HealthReport
{
    float bpm;   // 心率 (Beats Per Minute)
    float spo2;  // 血氧飽和度 (%)
    bool isBeat; // 本次採樣有沒有剛好抓到「心跳跳動的瞬間」
};

// 宣告處理演算法的函式
HealthReport process_PPG(uint32_t red, uint32_t ir);

#endif