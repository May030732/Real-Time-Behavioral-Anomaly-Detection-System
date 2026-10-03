class EDADetector:
    def __init__(self, baseline_sample_count=30, suspected_threshold=0.20, high_threshold=0.50, required_abnormal_count=3):
        self.baseline_sample_count = baseline_sample_count
        self.suspected_threshold = suspected_threshold
        self.high_threshold = high_threshold
        self.required_abnormal_count = required_abnormal_count
        
        self.baseline_values = []
        self.baseline_eda = None
        self.suspected_count = 0
        self.high_count = 0

    def detect_eda(self, current_eda):
        # 1. 排除無效 EDA
        if current_eda is None:
            print("EDA: None | Invalid EDA (Skipped)")
            return "Invalid_EDA"

        if current_eda <= 0:
            print(f"EDA: {current_eda:.3f} | Invalid EDA (Skipped)")
            return "Invalid_EDA"

        # 2. 建立個人 Baseline
        if self.baseline_eda is None:
            self.baseline_values.append(current_eda)
            print(
                f"Collecting baseline: "
                f"{len(self.baseline_values)}/{self.baseline_sample_count}"
                f" | EDA: {current_eda:.3f}"
            )

            if len(self.baseline_values) >= self.baseline_sample_count:
                self.baseline_eda = sum(self.baseline_values) / len(self.baseline_values)
                print("\n============================")
                print(f"EDA Baseline: {self.baseline_eda:.3f}")
                print("EDA Baseline completed")
                print("============================\n")

            return "Collecting_Baseline"

        # 3. 計算 EDA 與 Baseline 的差異
        eda_change = current_eda - self.baseline_eda
        change_ratio = eda_change / self.baseline_eda

        # 4. High Arousal
        if change_ratio >= self.high_threshold:
            self.high_count += 1
            self.suspected_count = 0
            if self.high_count >= self.required_abnormal_count:
                status = "High_Arousal"
            else:
                status = "Normal"

        # 5. Suspected Arousal
        elif change_ratio >= self.suspected_threshold:
            self.suspected_count += 1
            self.high_count = 0
            if self.suspected_count >= self.required_abnormal_count:
                status = "Suspected_Arousal"
            else:
                status = "Normal"

        # 6. Normal
        else:
            self.suspected_count = 0
            self.high_count = 0
            status = "Normal"

        # 7. 顯示判斷結果
        print(
            f"EDA: {current_eda:.3f}"
            f" | Baseline: {self.baseline_eda:.3f}"
            f" | Change: {eda_change:+.3f}"
            f" | Ratio: {change_ratio * 100:+.1f}%"
            f" | Suspected Count: {self.suspected_count}"
            f" | High Count: {self.high_count}"
            f" | Status: {status}"
        )

        return status