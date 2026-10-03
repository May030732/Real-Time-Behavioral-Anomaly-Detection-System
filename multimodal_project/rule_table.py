# ==============================================================================
# rule_table.py
# PPG、EDA、IMU 共 36 種狀態組合查表與分類器
# ==============================================================================

DECISION_RULE_TABLE = {
    # ---------------- 1~9. PPG: NORMAL ----------------
    ("NORMAL", "Normal", "calm"): "Normal",                         # 1
    ("NORMAL", "Normal", "slightly_intense"): "Normal",             # 2
    ("NORMAL", "Normal", "fierce"): "Normal",                       # 3
    ("NORMAL", "Suspected_Arousal", "calm"): "Normal",               # 4
    ("NORMAL", "Suspected_Arousal", "slightly_intense"): "Normal",   # 5
    ("NORMAL", "Suspected_Arousal", "fierce"): "Normal",             # 6
    ("NORMAL", "High_Arousal", "calm"): "Suspected_Abnormal",         # 7
    ("NORMAL", "High_Arousal", "slightly_intense"): "Suspected_Abnormal",  # 8
    ("NORMAL", "High_Arousal", "fierce"): "Suspected_Abnormal",       # 9

    # ---------------- 10~18. PPG: PERSISTENT_HIGH_HR ----------------
    ("PERSISTENT_HIGH_HR", "Normal", "calm"): "Suspected_Abnormal",   # 10
    ("PERSISTENT_HIGH_HR", "Normal", "slightly_intense"): "Normal",  # 11
    ("PERSISTENT_HIGH_HR", "Normal", "fierce"): "Normal",            # 12
    ("PERSISTENT_HIGH_HR", "Suspected_Arousal", "calm"): "Suspected_Abnormal",  # 13
    ("PERSISTENT_HIGH_HR", "Suspected_Arousal", "slightly_intense"): "Normal",  # 14
    ("PERSISTENT_HIGH_HR", "Suspected_Arousal", "fierce"): "Normal", # 15
    ("PERSISTENT_HIGH_HR", "High_Arousal", "calm"): "Suspected_Abnormal",  # 16
    ("PERSISTENT_HIGH_HR", "High_Arousal", "slightly_intense"): "Suspected_Abnormal",  # 17
    ("PERSISTENT_HIGH_HR", "High_Arousal", "fierce"): "Suspected_Abnormal",  # 18

    # ---------------- 19~27. PPG: SUDDEN_HR_CHANGE ----------------
    ("SUDDEN_HR_CHANGE", "Normal", "calm"): "Abnormal",              # 19
    ("SUDDEN_HR_CHANGE", "Normal", "slightly_intense"): "Suspected_Abnormal",  # 20
    ("SUDDEN_HR_CHANGE", "Normal", "fierce"): "Suspected_Abnormal",   # 21
    ("SUDDEN_HR_CHANGE", "Suspected_Arousal", "calm"): "Suspected_Abnormal",  # 22
    ("SUDDEN_HR_CHANGE", "Suspected_Arousal", "slightly_intense"): "Suspected_Abnormal",  # 23
    ("SUDDEN_HR_CHANGE", "Suspected_Arousal", "fierce"): "Suspected_Abnormal",  # 24
    ("SUDDEN_HR_CHANGE", "High_Arousal", "calm"): "Suspected_Abnormal",  # 25
    ("SUDDEN_HR_CHANGE", "High_Arousal", "slightly_intense"): "Suspected_Abnormal",  # 26
    ("SUDDEN_HR_CHANGE", "High_Arousal", "fierce"): "Suspected_Abnormal",  # 27

    # ---------------- 28~36. PPG: CRITICAL_EXTREME_HR ----------------
    ("CRITICAL_EXTREME_HR", "Normal", "calm"): "Abnormal",           # 28
    ("CRITICAL_EXTREME_HR", "Normal", "slightly_intense"): "Abnormal",  # 29
    ("CRITICAL_EXTREME_HR", "Normal", "fierce"): "Abnormal",         # 30
    ("CRITICAL_EXTREME_HR", "Suspected_Arousal", "calm"): "Abnormal",  # 31
    ("CRITICAL_EXTREME_HR", "Suspected_Arousal", "slightly_intense"): "Abnormal",  # 32
    ("CRITICAL_EXTREME_HR", "Suspected_Arousal", "fierce"): "Abnormal",  # 33
    ("CRITICAL_EXTREME_HR", "High_Arousal", "calm"): "Abnormal",     # 34
    ("CRITICAL_EXTREME_HR", "High_Arousal", "slightly_intense"): "Abnormal",  # 35
    ("CRITICAL_EXTREME_HR", "High_Arousal", "fierce"): "Abnormal",   # 36
}


def get_final_decision(
    ppg_state: str,
    eda_state: str,
    imu_state: str,
) -> str:
    """
    根據三種感測器狀態，查表取得手環判斷結果。

    PPG:
        NORMAL
        PERSISTENT_HIGH_HR
        SUDDEN_HR_CHANGE
        CRITICAL_EXTREME_HR

    EDA:
        Normal
        Suspected_Arousal
        High_Arousal

    IMU:
        calm
        slightly_intense
        fierce

    回傳:
        Normal / Suspected_Abnormal / Abnormal
        未定義的輸入則回傳 Unknown
    """
    key = (ppg_state, eda_state, imu_state)
    result = DECISION_RULE_TABLE.get(key, "Unknown")

    if result == "Unknown":
        print(
            "⚠️ [Rule Table 警告] 傳入未定義的組合狀態："
            f"PPG={ppg_state}, EDA={eda_state}, IMU={imu_state}"
        )

    return result


if __name__ == "__main__":
    from itertools import product
    from collections import Counter

    ppg_states = (
        "NORMAL",
        "PERSISTENT_HIGH_HR",
        "SUDDEN_HR_CHANGE",
        "CRITICAL_EXTREME_HR",
    )
    eda_states = ("Normal", "Suspected_Arousal", "High_Arousal")
    imu_states = ("calm", "slightly_intense", "fierce")

    expected_keys = set(product(ppg_states, eda_states, imu_states))

    # 確認沒有漏掉組合或加入錯誤的狀態名稱
    assert set(DECISION_RULE_TABLE) == expected_keys, "36 種組合不完整"

    counts = Counter(DECISION_RULE_TABLE.values())
    assert counts == {
        "Normal": 10,
        "Suspected_Abnormal": 16,
        "Abnormal": 10,
    }, f"分類數量不符：{counts}"

    print("=== 測試全部 36 種狀態組合 ===")

    for index, states in enumerate(
        product(ppg_states, eda_states, imu_states), start=1
    ):
        result = get_final_decision(*states)
        print(f"{index:2d}. {states} → {result}")

    assert get_final_decision("Unknown", "Normal", "calm") == "Unknown"
    print("\n✅ 組合完整性、分類數量及未知輸入檢查通過")