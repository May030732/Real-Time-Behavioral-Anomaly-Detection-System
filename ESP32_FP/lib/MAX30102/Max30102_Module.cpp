#include "Max30102_Module.h"
#include <Wire.h>

// 原廠硬體規定死的身分證字號 (I2C Address)
const int MAX_addr = 0x57;

// 常用暫存器門牌定義
const byte REG_FIFO_DATA = 0x07;
const byte REG_MODE_CONFIG = 0x09;
const byte REG_SPO2_CONFIG = 0x0A;
const byte REG_LED1_PA = 0x0C; // 紅光 LED 電流
const byte REG_LED2_PA = 0x0D; // 紅外線 LED 電流

void init_MAX30102()
{
    // 1. 軟體復位 (Reset) 晶片
    Wire.beginTransmission(MAX_addr);
    Wire.write(REG_MODE_CONFIG);
    Wire.write(0x40); // 寫入 0x40 觸發 Reset
    Wire.endTransmission();
    delay(100);

    // 2. 設定模式：同時開啟 紅光 與 紅外線 LED (SpO2 模式)
    Wire.beginTransmission(MAX_addr);
    Wire.write(REG_MODE_CONFIG);
    Wire.write(0x03);
    Wire.endTransmission();

    // 3. 設定 ADC 範圍與採樣率 (控制解析度)
    Wire.beginTransmission(MAX_addr);
    Wire.write(REG_SPO2_CONFIG);
    Wire.write(0x27); // 18位元 ADC 解析度, 100Hz 採樣率
    Wire.endTransmission();

    // 4. 設定 LED 的發光強度 (電流大小)
    Wire.beginTransmission(MAX_addr);
    Wire.write(REG_LED1_PA);
    Wire.write(0x24); // 紅光電流設為約 7.2mA
    Wire.endTransmission();

    Wire.beginTransmission(MAX_addr);
    Wire.write(REG_LED2_PA);
    Wire.write(0x24); // 紅外線電流設為約 7.2mA
    Wire.endTransmission();

    Serial.println("[Module] MAX30102 暫存器配置完成！");
}

// 檢查 FIFO 緩衝區裡面有沒有新資料可以讀取
bool available_MAX30102()
{
    // 讀取暫存器 0x04 (FIFO Write Pointer) 與 0x06 (FIFO Read Pointer)
    // 為了簡化專題複雜度，此處採用定時查詢，直接回傳 true
    return true;
}

HeartRateData read_MAX30102()
{
    HeartRateData data = {0, 0};

    // 1. 指定讀取起點為 FIFO 數據暫存器 (0x07)
    Wire.beginTransmission(MAX_addr);
    Wire.write(REG_FIFO_DATA);
    Wire.endTransmission(false);

    // 2. 請求 6 個位元組：紅光佔 3 bytes，紅外線佔 3 bytes
    Wire.requestFrom(MAX_addr, 6, true);

    if (Wire.available() == 6)
    {
        // 3. 讀取紅光 24 位元資料 (連續抓 3 次 8 位元箱子進行位移拼接)
        data.red = (uint32_t)Wire.read() << 16;
        data.red |= (uint32_t)Wire.read() << 8;
        data.red |= Wire.read();
        data.red &= 0x03FFFF; // MAX30102 最大解析度為 18 位元，清除高位多餘雜訊

        // 4. 讀取紅外線 24 位元資料
        data.ir = (uint32_t)Wire.read() << 16;
        data.ir |= (uint32_t)Wire.read() << 8;
        data.ir |= Wire.read();
        data.ir &= 0x03FFFF;
    }

    return data;
}