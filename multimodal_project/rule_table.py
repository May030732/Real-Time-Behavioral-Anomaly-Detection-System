# ==============================================================================
# rule_table.py
# 三模態 (PPG, EDA, IMU) 27 種狀態組合查表與分類器
# ==============================================================================

# 27 種排列組合對照表 (Key: (PPG, EDA, IMU), Value: 最終結果)
DECISION_RULE_TABLE = {
    # ---------------- 1~9. PPG: Slow ----------------
    ("Slow", "Normal", "calm"): "Normal",                             # 1
    ("Slow", "Normal", "slightly_intense"): "Normal",                   # 2
    ("Slow", "Normal", "fierce"): "Suspected_Abnormal",                 # 3
    ("Slow", "Suspected_Arousal", "calm"): "Suspected_Abnormal",        # 4
    ("Slow", "Suspected_Arousal", "slightly_intense"): "Suspected_Abnormal", # 5
    ("Slow", "Suspected_Arousal", "fierce"): "Abnormal",               # 6
    ("Slow", "High_Arousal", "calm"): "Suspected_Abnormal",             # 7
    ("Slow", "High_Arousal", "slightly_intense"): "Abnormal",           # 8
    ("Slow", "High_Arousal", "fierce"): "Abnormal",                   # 9

    # ---------------- 10~18. PPG: Medium ----------------
    ("Medium", "Normal", "calm"): "Normal",                           # 10
    ("Medium", "Normal", "slightly_intense"): "Normal",                 # 11
    ("Medium", "Normal", "fierce"): "Suspected_Abnormal",               # 12
    ("Medium", "Suspected_Arousal", "calm"): "Suspected_Abnormal",       # 13
    ("Medium", "Suspected_Arousal", "slightly_intense"): "Suspected_Abnormal", # 14
    ("Medium", "Suspected_Arousal", "fierce"): "Abnormal",             # 15
    ("Medium", "High_Arousal", "calm"): "Suspected_Abnormal",           # 16
    ("Medium", "High_Arousal", "slightly_intense"): "Abnormal",         # 17
    ("Medium", "High_Arousal", "fierce"): "Abnormal",                 # 18

    # ---------------- 19~27. PPG: Fast ----------------
    ("Fast", "Normal", "calm"): "Suspected_Abnormal",                   # 19
    ("Fast", "Normal", "slightly_intense"): "Suspected_Abnormal",       # 20
    ("Fast", "Normal", "fierce"): "Abnormal",                         # 21
    ("Fast", "Suspected_Arousal", "calm"): "Suspected_Abnormal",        # 22
    ("Fast", "Suspected_Arousal", "slightly_intense"): "Abnormal",       # 23
    ("Fast", "Suspected_Arousal", "fierce"): "Abnormal",               # 24
    ("Fast", "High_Arousal", "calm"): "Abnormal",                       # 25
    ("Fast", "High_Arousal", "slightly_intense"): "Abnormal",           # 26
    ("Fast", "High_Arousal", "fierce"): "Abnormal",                     # 27
}


def get_final_decision(ppg_state: str, eda_state: str, imu_state: str) -> str:
    """
    輸入三種感測器的單獨狀態文字，經由 27 種真值表映射出最終系統狀態
    
    :param ppg_state: 'Slow', 'Medium', 'Fast'
    :param eda_state: 'Normal', 'Suspected_Arousal', 'High_Arousal'
    :param imu_state: 'calm', 'slightly_intense', 'fierce'
    :return: 'Normal', 'Suspected_Abnormal', 'Abnormal'
    """
    key = (ppg_state, eda_state, imu_state)
    
    # 查表，若輸入不存在的狀態字串則回傳 Unknown
    result = DECISION_RULE_TABLE.get(key, "Unknown")
    
    if result == "Unknown":
        print(f"⚠️ [Rule Table 警告] 傳入未定義的組合狀態: PPG={ppg_state}, EDA={eda_state}, IMU={imu_state}")
        
    return result


# 模組測試用程式區塊
if __name__ == "__main__":
    print("=== 測試 27 種狀態組合查表 ===")
    test_cases = [
        ("Slow", "Normal", "calm"),               # 應輸出 Normal
        ("Medium", "Suspected_Arousal", "calm"),  # 應輸出 Suspected_Abnormal
        ("Fast", "Suspected_Arousal", "fierce"),   # 應輸出 Abnormal
    ]
    
    for ppg, eda, imu in test_cases:
        res = get_final_decision(ppg, eda, imu)
        print(f"PPG: {ppg:6s} | EDA: {eda:17s} | IMU: {imu:16s} ➔ 判定: {res}")