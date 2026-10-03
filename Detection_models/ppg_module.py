from collections import deque

class PPGDetector:
    def __init__(
        self,
        window_size_sec=15,
        extreme_low_bpm=40,
        extreme_high_bpm=180,
        required_extreme_windows=2,  # 2 個視窗 = 30 秒
        spike_drop_ratio=0.4,        # 偏離 2 分鐘均值 40%
        min_2min_buffer=4,           # 至少 1 分鐘 (4 個視窗) 歷史才比對劇變
        max_2min_buffer=8,           # 最多 2 分鐘 (8 個視窗)
        persistent_high_bpm=100,
        required_high_windows=4      # 4 個視窗 = 1 分鐘
    ):
        self.window_size_sec = window_size_sec
        self.extreme_low_bpm = extreme_low_bpm
        self.extreme_high_bpm = extreme_high_bpm
        self.required_extreme_windows = required_extreme_windows
        self.spike_drop_ratio = spike_drop_ratio
        self.min_2min_buffer = min_2min_buffer
        self.persistent_high_bpm = persistent_high_bpm
        self.required_high_windows = required_high_windows

        # 視窗緩衝區與歷史狀態
        self.current_window_samples = []
        self.history_2min_windows = deque(maxlen=max_2min_buffer)
        self.extreme_status_history = deque(maxlen=required_extreme_windows)
        self.high_bpm_history = deque(maxlen=required_high_windows)

    def process_sample(self, current_bpm):
        """
        接收每秒傳入的 BPM 採樣值。
        未集滿 15 秒回傳 None；滿 15 秒時計算均值並執行三層評估。
        """
        if current_bpm is None or current_bpm <= 0:
            return None

        self.current_window_samples.append(current_bpm)

        # 累積滿 15 秒樣本數
        if len(self.current_window_samples) >= self.window_size_sec:
            avg_bpm = sum(self.current_window_samples) / len(self.current_window_samples)
            self.current_window_samples.clear()
            status, window_avg = self._evaluate_window(avg_bpm)
            return status, window_avg

        return None

    def _evaluate_window(self, window_bpm):
        """三層核心檢測邏輯"""
        # 第一層：極限值判斷 (<40 或 >180，持續 30 秒)
        is_extreme = (window_bpm < self.extreme_low_bpm) or (window_bpm > self.extreme_high_bpm)
        self.extreme_status_history.append(is_extreme)

        if len(self.extreme_status_history) == self.required_extreme_windows and all(self.extreme_status_history):
            self.history_2min_windows.append(window_bpm)
            return "CRITICAL_EXTREME_HR", window_bpm

        # 第二層：突發性劇變 (較前 2 分鐘均值變化 > 40%)
        if len(self.history_2min_windows) >= self.min_2min_buffer:
            baseline_avg = sum(self.history_2min_windows) / len(self.history_2min_windows)
            upper_bound = baseline_avg * (1 + self.spike_drop_ratio)
            lower_bound = baseline_avg * (1 - self.spike_drop_ratio)

            if window_bpm >= upper_bound or window_bpm <= lower_bound:
                self.history_2min_windows.append(window_bpm)
                return "SUDDEN_HR_CHANGE", window_bpm

        # 第三層：持續性偏高 (>100，持續 1 分鐘)
        is_high = window_bpm > self.persistent_high_bpm
        self.high_bpm_history.append(is_high)

        if len(self.high_bpm_history) == self.required_high_windows and all(self.high_bpm_history):
            status = "PERSISTENT_HIGH_HR"
        else:
            status = "NORMAL"

        self.history_2min_windows.append(window_bpm)
        return status, window_bpm