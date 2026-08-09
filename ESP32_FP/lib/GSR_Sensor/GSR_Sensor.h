#ifndef GSR_SENSOR_H
#define GSR_SENSOR_H

#include <Arduino.h>

class GSR_Sensor
{
private:
    uint8_t _pin;           // 連接的 ADC 腳位 (例如 GPIO34)
    uint16_t _lastRawValue; // 儲存最近一次讀取到的原始 ADC 值
    float _filteredValue;   // 儲存濾波後的數值
    float _smoothFactor;    // 低通濾波係數 (0.0 ~ 1.0)

public:
    /**
     * @brief 建構子
     * @param pin 連接 ESP32 的 ADC 腳位 (建議使用 IO34, IO35 等 ADC1 腳位)
     * @param smoothFactor 濾波平滑度 (預設 0.2，越小越平滑但反應變慢)
     */
    GSR_Sensor(uint8_t pin, float smoothFactor = 0.2);

    /**
     * @brief 初始化 GSR 感測器設定
     */
    void begin();

    /**
     * @brief 讀取一次原始 ADC 值 (0 ~ 4095)
     * @return uint16_t 原始數值
     */
    uint16_t readRaw();

    /**
     * @brief 讀取經過低通濾波後的平滑數值 (適合觀察情緒/皮膚電反應趨勢)
     * @return float 平滑後的數值
     */
    float readSmooth();

    /**
     * @brief 取得簡單的微西門子 (uS) 電導度估算值 (選用)
     * @return float 電導度 (uS)
     */
    float readConductance();
};

#endif // GSR_SENSOR_H