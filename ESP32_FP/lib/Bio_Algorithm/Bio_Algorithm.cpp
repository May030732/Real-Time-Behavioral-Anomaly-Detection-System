#include "Bio_Algorithm.h"

// 演算法內部使用的濾波與暫存變數
static float ir_dc = 130000.0;
static float red_dc = 115000.0;
static uint32_t last_beat_time = 0;

// 血氧與動態門檻所需的變數
static float ir_max = -9999.0, ir_min = 999999.0;
static float red_max = -9999.0, red_min = 999999.0;
static float last_peak_to_peak = 100.0; // 記錄上一次的波對波振幅

// 儲存最新算出的數值，維持持續輸出
static float current_bpm = 0.0;
static float current_spo2 = 0.0;

HealthReport process_PPG(uint32_t red, uint32_t ir)
{
    HealthReport report = {current_bpm, current_spo2, false};

    // 1. 手指未按壓或脫離：重置所有演算法狀態
    if (ir < 30000 || red < 30000)
    {
        ir_max = -9999;
        ir_min = 999999;
        red_max = -9999;
        red_min = 999999;
        last_beat_time = 0;
        current_bpm = 0.0;
        current_spo2 = 0.0;
        return report;
    }

    uint32_t now = millis();

    // 2. 低通濾波：動態追蹤基底 DC 值
    ir_dc = ir_dc * 0.95 + (float)ir * 0.05;
    red_dc = red_dc * 0.95 + (float)red * 0.05;

    // 3. 提取微小的心跳波動訊號 (AC)
    float ir_ac = (float)ir - ir_dc;
    float red_ac = (float)red - red_dc;

    // 4. 追蹤極值 (用於計算 AC 振幅)
    if (ir_ac > ir_max)
        ir_max = ir_ac;
    if (ir_ac < ir_min)
        ir_min = ir_ac;
    if (red_ac > red_max)
        red_max = red_ac;
    if (red_ac < red_min)
        red_min = red_ac;

    // 5. 超時自動重置機制 (解開 Deadlock 的關鍵！)
    // 必須放在心跳判斷外層：如果超過 1.5 秒都沒有捕捉到心跳，強制降門檻重新搜尋
    if (last_beat_time > 0 && (now - last_beat_time) >= 1500)
    {
        last_beat_time = now;
        last_peak_to_peak = 50.0; // 強制壓低動態門檻，迎合弱訊號

        ir_max = -9999;
        ir_min = 999999;
        red_max = -9999;
        red_min = 999999;
    }

    // 6. 動態計算心跳判斷門檻
    float dynamic_threshold = last_peak_to_peak * 0.3;
    if (dynamic_threshold < 20.0)
        dynamic_threshold = 20.0; // 最低雜訊下限

    // 7. 心率捕捉邏輯 (負向穿過門檻點)
    static float last_ir_ac = 0;
    if (last_ir_ac > dynamic_threshold && ir_ac <= dynamic_threshold)
    {
        if (last_beat_time == 0)
        {
            last_beat_time = now;
        }
        else
        {
            uint32_t delta = now - last_beat_time;

            // 合理心跳區間：350ms ~ 1500ms (40 ~ 170 BPM)
            if (delta >= 350 && delta < 1500)
            {
                float raw_bpm = 60000.0 / (float)delta;

                // 軟體平滑濾波 (改為 75% 歷史 + 25% 新值，保留自然 RR 波動度，不會顯得太死板)
                if (current_bpm == 0.0)
                    current_bpm = raw_bpm;
                else
                    current_bpm = current_bpm * 0.75 + raw_bpm * 0.25;

                // 計算血氧 (SpO2)
                float ir_signal_amplitude = ir_max - ir_min;
                float red_signal_amplitude = red_max - red_min;

                if (ir_signal_amplitude > 15.0 && red_signal_amplitude > 15.0)
                {
                    last_peak_to_peak = ir_signal_amplitude; // 更新下一次動態門檻基準

                    float R = (red_signal_amplitude / red_dc) / (ir_signal_amplitude / ir_dc);

                    // 標準 R-Curve 經驗公式，微調參數使其更自然
                    float calculated_spo2 = 104.0 - 17.0 * R;

                    if (calculated_spo2 > 100.0)
                        calculated_spo2 = 100.0;
                    if (calculated_spo2 < 80.0)
                        calculated_spo2 = 80.0;

                    // SpO2 也加入輕微平滑，避免突變
                    if (current_spo2 == 0.0)
                        current_spo2 = calculated_spo2;
                    else
                        current_spo2 = current_spo2 * 0.8 + calculated_spo2 * 0.2;
                }

                // 重置極值，開始下一個心跳週期的統計
                ir_max = -9999;
                ir_min = 999999;
                red_max = -9999;
                red_min = 999999;

                last_beat_time = now;
                report.isBeat = true; // 標記此採樣點觸發了一次心跳
            }
        }
    }

    last_ir_ac = ir_ac;

    // 將最新算出的 BPM 與 SpO2 帶入報告中輸出
    report.bpm = current_bpm;
    report.spo2 = current_spo2;

    return report;
}