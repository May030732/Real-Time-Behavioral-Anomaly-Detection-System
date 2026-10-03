# 穿戴式多模態生理與行為異常偵測系統規格書 (Multimodal Anomaly Detection Specification)

本文件定義系統中 **IMU（慣性感測器）**、**PPG（光學脈搏訊號）** 與 **EDA/GSR（皮膚電阻/電導訊號）** 三個模組之輸入資料型態、演算法判斷機制、模組狀態輸出，以及多模態決策融合（Decision Fusion）規則。

---

## 1. 各感測模組輸入與判定邏輯

### 1.1 IMU 動作行為強度模組 (`IMUModule`)

* **輸入特徵 (Input)**:
  * 原始六軸時間序列資料視窗：`120` 筆（採樣率 60Hz，時間長度 2 秒）。
  * 欄位包含三軸加速度 $a_x, a_y, a_z$ 與三軸角速度 $g_x, g_y, g_z$。
  * 內部萃取 8 條訊號（包含向量合成量 $acc_{mag}, gyr_{mag}$）之時域特徵：平均值 (Mean)、標準差 (Std)、極值範圍 (Range)、均方根 (RMS)、能量 (Energy) 等。
* **分類模型**:
  * 預訓練隨機森林分類器 (`layout25_intensity_random_forest_v1.joblib`)。
* **輸出狀態 (Output)**:
  * `calm`: 平靜 / 靜態或微幅肢體活動。
  * `slightly_intense`: 中等強度動作（如輕度走動、手部微幅擺動）。
  * `fierce`: 激烈動作（如快跑、大幅揮手或強烈衝擊）。

---

### 1.2 PPG 心率異常分層檢測模組 (`PPGModule`)

* **輸入特徵 (Input)**:
  * 即時心率數值 `bpm`（以 1 秒為週期採樣）。
  * 採樣累積滿 15 筆後，計算 **15 秒視窗平均心率** ($BPM_{15s}$) 作為分層檢測依據。
* **分層判斷邏輯**:
  1. **第一層：極限值判斷 (Critical Extremes)**
     * 條件：$BPM_{15s} < 40$ 或 $BPM_{15s} > 180$。
     * 判定機制：使用 `extreme_status_history` 隊列記錄，需**連續 2 個視窗（累計 30 秒）**成立。
     * 輸出狀態：`CRITICAL_EXTREME_HR`
  2. **第二層：突發性劇變 (Sudden Spike / Drop)**
     * 條件：當前 $BPM_{15s}$ 偏離前 2 分鐘歷史平均基準達 $\pm 40\%$。
     * 基準計算：維護最多 8 個視窗（共 2 分鐘）的滑動均值 $BPM_{2min\_avg}$，且歷史視窗需 $\ge 4$（至少 1 分鐘資料）。
     * 門檻：$BPM_{15s} \ge 1.4 \times BPM_{2min\_avg}$ 或 $BPM_{15s} \le 0.6 \times BPM_{2min\_avg}$。
     * 輸出狀態：`SUDDEN_HR_CHANGE`
  3. **第三層：持續性偏高 (Persistent High)**
     * 條件：$BPM_{15s} > 100$。
     * 判定機制：使用 `high_bpm_history` 隊列記錄，需**連續 4 個視窗（累計 1 分鐘）**成立。
     * 輸出狀態：`PERSISTENT_HIGH_HR`
  4. **第四層：心率正常 (Normal Baseline)**
     * 條件：未觸發上述三層異常條件。
     * 輸出狀態：`NORMAL`
  5. **輔助狀態**:
     * `Collecting_Window`: 尚未累積滿 15 秒採樣資料。
     * `Invalid_BPM`: $BPM \le 0$ 或輸入值為空。

---

### 1.3 EDA / GSR 皮膚電導反應模組 (`EDAModule`)

* **輸入特徵 (Input)**:
  * 即時皮膚電阻 ADC 平滑值或微西門子電導度 `current_eda`。
* **判斷邏輯**:
  * 基準線建立：前 30 筆非零數據累積平均，計算出個人基準線 $EDA_{baseline}$。
  * 變化比例計算：
    $$\Delta Ratio = \frac{current\_eda - EDA_{baseline}}{EDA_{baseline}}$$
* **輸出狀態 (Output)**:
  * `Collecting_Baseline`: 前 30 筆資料收集與校正中。
  * `Invalid_EDA`: 輸入值為 None 或 $\le 0$。
  * `Normal`: 數值未達異常門檻，或異常次數尚未連續累積達標。
  * `Suspected_Arousal`: $\Delta Ratio \ge 20\%$，且連續出現 3 次以上。
  * `High_Arousal`: $\Delta Ratio \ge 50\%$，且連續出現 3 次以上。

---

## 2. 多感測器決策融合機制 (Decision Fusion)

系統在 `evaluate_fusion_decision(imu_st, ppg_st, eda_st)` 依循以下優先序與交叉規則產生最終決策（Final Decision）：

```
                                [開始決策評估]
                                       │
                    ┌──────────────────┴──────────────────┐
                    │ PPG == 'CRITICAL_EXTREME_HR'?       │
                    └──────────────────┬──────────────────┘
                                       │
                      [是] ────────────┴──────────── [否]
                       │                              │
                       ▼                              ▼
             Medical_Emergency         生理激發狀態判定:
                                       is_ppg_aroused = PPG in [SUDDEN_HR_CHANGE, PERSISTENT_HIGH_HR]
                                       is_eda_aroused = EDA in [Suspected_Arousal, High_Arousal]
                                       is_physiological_aroused = is_ppg_aroused OR is_eda_aroused
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
            [is_physiological_aroused 為 True]                           [is_physiological_aroused 為 False]
                       │                                                             │
            ┌──────────┴──────────┐                                                  ▼
            │ IMU 狀態為何？       │                                        IMU == 'fierce'?
            └──────────┬──────────┘                                                  │
                       │                                            ┌────────────────┴────────────────┐
          ┌────────────┴────────────┐                              [是]                              [否]
          ▼                         ▼                               ▼                                 ▼
   calm / slightly_intense        fierce                   High_Motion_Normal           檢查是否有 Calibrating/Invalid
          │                         │                                                             │
          ▼                         ▼                                                             ▼
   Emotional_Stress         Physical_Exertion                                                Normal / Warning
```

### 決策融合真值表 (Decision Matrix)

| 優先序 | IMU 狀態 | PPG 狀態 | EDA 狀態 | 最終輸出 (`final_decision`) | 結果含意說明 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **P1** | 任意狀態 | `CRITICAL_EXTREME_HR` | 任意狀態 | **`Medical_Emergency`** | **醫療緊急狀態**：連續 30 秒心率 $<40$ 或 $>180$，具高度心血管危機。 |
| **P2** | `calm` 或 `slightly_intense` | `SUDDEN_HR_CHANGE` 或 `PERSISTENT_HIGH_HR` | `Suspected_Arousal` 或 `High_Arousal` | **`Emotional_Stress`** | **情緒/心理壓力**：肢體無劇烈運動，但自主神經與心率明顯激發。 |
| **P3** | `fierce` | 異常或偏高 | 異常或偏高 | **`Physical_Exertion`** | **劇烈運動負荷**：肢體運動劇烈，生理指標上升屬於正常運動生理反應。 |
| **P4** | `fierce` | `NORMAL` | `Normal` | **`High_Motion_Normal`** | **高活動平穩狀態**：肢體劇烈活動中，心率與皮膚電導仍在耐受範圍內。 |
| **P5** | 任意狀態包含 `Collecting_*` | 任意狀態包含 `Collecting_*` | 任意狀態包含 `Collecting_*` | **`Calibrating`** | **系統校正中**：模組正在收集基線或視窗資料。 |
| **P6** | 任意狀態包含 `Invalid_*` | 任意狀態包含 `Invalid_*` | 任意狀態包含 `Invalid_*` | **`Sensor_Warning`** | **感測器異常**：接觸不良、未配戴或數值超出合理範圍。 |
| **P7** | `calm` 或 `slightly_intense` | `NORMAL` | `Normal` | **`Normal`** | **常態健康狀態**：各項生理與姿態特徵均在正常基準區間。 |

---

## 3. 資料庫儲存欄位定義 (`detection_result.db`)

融合判斷完成後，每筆結果會同步寫入 `results` 資料表：

| 欄位名稱 | 型態 | 範例值 | 說明 |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | `102` | 主鍵，自動遞增 |
| `timestamp` | TEXT | `2026-10-03 15:30:12` | 記錄時間點 |
| `latest_sensor_id` | INTEGER | `12500` | 對應 `sensor_data` 的最新資料流水號 |
| `imu_status` | TEXT | `calm` | IMU 動作強度輸出 |
| `ppg_bpm` | REAL | `108.4` | 即時量測心率 (BPM) |
| `ppg_status` | TEXT | `PERSISTENT_HIGH_HR` | PPG 分層檢測狀態碼 |
| `eda_value` | REAL | `1450.2` | 即時皮膚電阻 ADC / 電導值 |
| `eda_status` | TEXT | `Suspected_Arousal` | EDA 激發檢測狀態碼 |
| `final_decision` | TEXT | `Emotional_Stress` | 多模態決策融合最終結果 |