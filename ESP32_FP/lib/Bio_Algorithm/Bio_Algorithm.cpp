#include "Bio_Algorithm.h"

// 演算法內部使用的濾波與暫存變數
static float ir_dc = 130000.0;
static float red_dc = 115000.0;
static uint32_t last_beat_time = 0;

// 血氧與動態門檻所需的變數
static float ir_max = -9999.0, ir_min = 999999.0;
static float red_max = -9999.0, red_min = 999999.0;
static float last_peak_to_peak = 100.0; // 記錄上一次的波對波振幅

HealthReport process_PPG(uint32_t red, uint32_t ir)
{
    HealthReport report = {0.0, 0.0, false};

    // 如果手指根本沒壓，重置所有狀態
    if (ir < 30000 || red < 30000)
    {
        ir_max = -9999;
        ir_min = 999999;
        red_max = -9999;
        red_min = 999999;
        return report;
    }

    uint32_t now = millis();

    // 1. 低通濾波：動態追蹤基底 DC 值
    ir_dc = ir_dc * 0.98 + (float)ir * 0.02;
    red_dc = red_dc * 0.98 + (float)red * 0.02;

    // 2. 提取微小的心跳波動訊號 (AC)
    float ir_ac = (float)ir - ir_dc;
    float red_ac = (float)red - red_dc;

    // 3. 追蹤一個週期內的極大值與極小值
    if (ir_ac > ir_max)
        ir_max = ir_ac;
    if (ir_ac < ir_min)
        ir_min = ir_ac;
    if (red_ac > red_max)
        red_max = red_ac;
    if (red_ac < red_min)
        red_min = red_ac;

    // 4. 動態計算心跳判斷門檻 (取上一次振幅的 30% 作為動態觸發點)
    float dynamic_threshold = last_peak_to_peak * 0.3;
    if (dynamic_threshold < 30.0)
        dynamic_threshold = 30.0; // 設定最低雜訊下限

    // 5. 心率捕捉邏輯：當紅外線 AC 訊號越過動態波峰向下掉的瞬間
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

            // 【防禦防線 1】不應期保護：如果距離上一次心跳小於 350ms（相當於>170 BPM），判定為雜訊直接忽略！
            if (delta < 350)
            {
                // 視為血管微小回彈的次級波，不予理會
            }
            // 【防禦防線 2】合理心跳區間：0.35秒 ~ 1.5秒 (40 ~ 170 BPM)
            else if (delta >= 350 && delta < 1500)
            {
                float raw_bpm = 60000.0 / (float)delta;

                // 【防禦防線 3】軟體滑動平均濾波（讓數值更平滑，不會突變）
                static float filtered_bpm = 0.0;
                if (filtered_bpm == 0.0)
                    filtered_bpm = raw_bpm;
                else
                    filtered_bpm = filtered_bpm * 0.6 + raw_bpm * 0.4; // 60% 重視歷史，40% 引入新值

                report.bpm = filtered_bpm;
                report.isBeat = true;
                last_beat_time = now;

                // 計算血氧 (SpO2)
                float ir_signal_amplitude = ir_max - ir_min;
                float red_signal_amplitude = red_max - red_min;

                if (ir_signal_amplitude > 20 && red_signal_amplitude > 20)
                {
                    last_peak_to_peak = ir_signal_amplitude; // 更新下一次的動態門檻基準

                    float R = (red_signal_amplitude / red_dc) / (ir_signal_amplitude / ir_dc);
                    float calculated_spo2 = 110.0 - 25.0 * R;

                    if (calculated_spo2 > 100.0)
                        calculated_spo2 = 100.0;
                    if (calculated_spo2 < 75.0)
                        calculated_spo2 = 75.0;
                    report.spo2 = calculated_spo2;
                }

                // 重置極值
                ir_max = -9999;
                ir_min = 999999;
                red_max = -9999;
                red_min = 999999;
            }
            else if (delta >= 1500)
            {
                // 【終極修正】超時未觸發：代表目前門檻可能太高了，或病人手移開過
                last_beat_time = now;
                last_peak_to_peak = 100.0; // 強制將歷史振幅調小，也就是把下一次的門檻壓低，主動去迎合微弱的訊號

                // 同步重置極值追蹤，讓最大最小值重新累積
                ir_max = -9999;
                ir_min = 999999;
                red_max = -9999;
                red_min = 999999;
            }
        }
    }

    last_ir_ac = ir_ac;
    return report;
}