# ==========================================
# PPG BPM Arousal Detection
# BPM -> Baseline -> Abnormal Detection
# ==========================================


# 建立 baseline 要收集幾筆 BPM
BASELINE_SAMPLE_COUNT = 30

# 比 baseline 增加多少 BPM，先視為疑似 arousal
BPM_CHANGE_THRESHOLD = 25

# 必須連續幾次超過門檻
REQUIRED_ABNORMAL_COUNT = 3


# ==========================================
# 變數初始化
# ==========================================

baseline_values = []

baseline_bpm = None

abnormal_count = 0


# ==========================================
# 異常判斷函式
# ==========================================

def detect_arousal(current_bpm):

    global baseline_bpm
    global abnormal_count


    # --------------------------------------
    # 1. 排除無效 BPM
    # --------------------------------------

    if current_bpm <= 0:
        return "Invalid_BPM"


    # --------------------------------------
    # 2. 建立 Baseline
    # --------------------------------------

    if baseline_bpm is None:

        baseline_values.append(current_bpm)

        print(
            f"Collecting baseline: "
            f"{len(baseline_values)}/{BASELINE_SAMPLE_COUNT} "
            f"| BPM: {current_bpm:.1f}"
        )


        # 收集完成
        if len(baseline_values) >= BASELINE_SAMPLE_COUNT:

            baseline_bpm = sum(baseline_values) / len(baseline_values)

            print("\n============================")
            print(f"Baseline BPM: {baseline_bpm:.2f}")
            print("Baseline completed")
            print("============================\n")


        return "Collecting_Baseline"


    # --------------------------------------
    # 3. 計算 BPM Change
    # --------------------------------------

    bpm_change = current_bpm - baseline_bpm


    # --------------------------------------
    # 4. 判斷是否超過門檻
    # --------------------------------------

    if bpm_change >= BPM_CHANGE_THRESHOLD:

        abnormal_count += 1

    else:

        abnormal_count = 0


    # --------------------------------------
    # 5. 最終判斷
    # --------------------------------------

    if abnormal_count >= REQUIRED_ABNORMAL_COUNT:

        status = "Suspected_Arousal"

    else:

        status = "Normal"


    # --------------------------------------
    # 6. 顯示結果
    # --------------------------------------

    print(
        f"BPM: {current_bpm:.1f}"
        f" | Baseline: {baseline_bpm:.1f}"
        f" | Change: {bpm_change:+.1f}"
        f" | Count: {abnormal_count}"
        f" | Status: {status}"
    )


    return status