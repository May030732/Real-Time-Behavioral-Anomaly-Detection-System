#include "GSR_Sensor.h"

GSR_Sensor::GSR_Sensor(uint8_t pin, float smoothFactor)
{
    _pin = pin;
    _smoothFactor = smoothFactor;
    _lastRawValue = 0;
    _filteredValue = 0.0;
}

void GSR_Sensor::begin()
{
    pinMode(_pin, INPUT);

    // 設定 ESP32 ADC 解析度為 12-bit (0 ~ 4095)
    analogReadResolution(12);

    // 設定 ADC 衰減度，讓輸入範圍擴展至 0 ~ 3.3V
    analogSetAttenuation(ADC_11db);

    // 初始化第一次讀取
    _lastRawValue = analogRead(_pin);
    _filteredValue = (float)_lastRawValue;
}

uint16_t GSR_Sensor::readRaw()
{
    // 進行 5 次小采樣取平均，消除瞬間高頻電磁雜訊
    uint32_t sum = 0;
    for (int i = 0; i < 5; i++)
    {
        sum += analogRead(_pin);
        delayMicroseconds(200);
    }
    _lastRawValue = sum / 5;
    return _lastRawValue;
}

float GSR_Sensor::readSmooth()
{
    uint16_t currentRaw = readRaw();
    // 指數平滑低通濾波 (Exponential Moving Average Filter)
    _filteredValue = (_smoothFactor * currentRaw) + ((1.0 - _smoothFactor) * _filteredValue);
    return _filteredValue;
}

float GSR_Sensor::readConductance()
{
    uint16_t raw = readRaw();
    if (raw == 0)
        return 0.0;

    // 將 12-bit ADC (0-4095) 轉換為估算的皮膚電導度 (微西門子 uS)
    // 假設 ESP32 3.3V 供電，計算公式：
    float voltage = (raw / 4095.0) * 3.3;
    // 皮膚電阻估算 (歐姆) = (3.3 - Voltage) / Voltage * 分壓電阻(約100k)
    // 轉換成電導度 (uS) = 1,000,000 / Resistance
    if (3.3 - voltage <= 0.001)
        return 0.0; // 防止除以零

    float resistance = ((3.3 - voltage) / voltage) * 100000.0; // 100k ohm
    if (resistance <= 0)
        return 0.0;

    float uS = 1000000.0 / resistance;
    return uS;
}