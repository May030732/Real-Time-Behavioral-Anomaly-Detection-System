import mne
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# 開啟互動模式，讓五張圖可以同時顯示
plt.ion()


# ============================================================
# 1. 基本設定
# ============================================================

file_path = Path(
    r"C:\Users\lin shan mei\Desktop\專題暑期進度\scientisst"
    r"\scientisst-move-annotated-wearable-multimodal-biosignals-recorded-during-everyday-life-activities-in-naturalistic-environments-1.0.1"
    r"\03FH\empatica.edf"
)

# 完整 EDA 處理結果
processed_output_path = (
    file_path.parent / "eda_processed_data.csv"
)

# 每個時間窗的異常判斷結果
anomaly_output_path = (
    file_path.parent / "eda_anomaly_windows.csv"
)

# 顯示前幾秒
plot_duration_sec = 60

# 排除錄製開始的感測器初始化區段
ignore_first_sec = 10

# 每個分析時間窗長度
window_sec = 10

# 用排除初始化後的前幾秒建立基線
baseline_duration_sec = 120


# ============================================================
# 2. 檢查 EDF 檔案是否存在
# ============================================================

if not file_path.exists():
    raise FileNotFoundError(
        f"找不到 EDF 檔案：\n{file_path}\n"
        "請確認路徑與檔名是否正確。"
    )


# ============================================================
# 3. 讀取 EDF
# ============================================================

raw = mne.io.read_raw_edf(
    file_path,
    preload=True,
    verbose=False
)

print("=" * 60)
print("EDF 讀取成功")
print("所有通道：")
print(raw.ch_names)
print("=" * 60)


# ============================================================
# 4. 取得 EDA 通道
# ============================================================

eda_channel = "eda:dry"

if eda_channel not in raw.ch_names:
    raise ValueError(
        f"找不到通道：{eda_channel}\n"
        f"目前可使用的通道：{raw.ch_names}"
    )

eda_data = raw.get_data(
    picks=[eda_channel]
)[0]

sampling_rate = float(
    raw.info["sfreq"]
)

time = raw.times


print("EDA 通道：", eda_channel)
print("EDA 資料筆數：", len(eda_data))
print("EDA 取樣率：", sampling_rate, "Hz")
print(
    "EDA 總長度：",
    len(eda_data) / sampling_rate,
    "秒"
)
print("EDA 前 10 筆：", eda_data[:10])


# ============================================================
# 5. 處理 NaN 或無限值
# ============================================================

invalid_count = np.sum(
    ~np.isfinite(eda_data)
)

if invalid_count > 0:
    print(
        f"發現 {invalid_count} 筆 NaN 或無限值，"
        "將使用線性插值處理。"
    )

    eda_series = pd.Series(
        eda_data
    )

    eda_data = (
        eda_series
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .interpolate(
            method="linear"
        )
        .bfill()
        .ffill()
        .to_numpy()
    )


# ============================================================
# 6. 使用 NeuroKit2 處理 EDA
# ============================================================

print("\n開始處理 EDA，請稍候……")

signals, info = nk.eda_process(
    eda_data,
    sampling_rate=sampling_rate,
    method="neurokit"
)

print("EDA 處理完成。")

print("\nNeuroKit2 產生的欄位：")

for column in signals.columns:
    print("-", column)


# ============================================================
# 7. 確認必要欄位存在
# ============================================================

required_columns = [
    "EDA_Raw",
    "EDA_Clean",
    "EDA_Tonic",
    "EDA_Phasic",
    "SCR_Onsets",
    "SCR_Peaks",
    "SCR_Amplitude"
]

missing_columns = [
    column
    for column in required_columns
    if column not in signals.columns
]

if missing_columns:
    raise KeyError(
        "缺少以下 NeuroKit2 欄位："
        + ", ".join(missing_columns)
    )


# ============================================================
# 8. 轉成 NumPy 陣列
# ============================================================

eda_raw = signals[
    "EDA_Raw"
].to_numpy()

eda_clean = signals[
    "EDA_Clean"
].to_numpy()

scl = signals[
    "EDA_Tonic"
].to_numpy()

scr_phasic = signals[
    "EDA_Phasic"
].to_numpy()

scr_peaks = signals[
    "SCR_Peaks"
].to_numpy()

scr_amplitude = signals[
    "SCR_Amplitude"
].to_numpy()


# ============================================================
# 9. 找出完整資料中的 SCR Peaks
# ============================================================

peak_indices = np.where(
    scr_peaks == 1
)[0]


# ============================================================
# 10. 只分析前 60 秒
# ============================================================

analysis_duration_sec = 60
ignore_first_sec = 10
window_sec = 10
baseline_duration_sec = 20

ignore_samples = int(
    ignore_first_sec * sampling_rate
)

window_samples = int(
    window_sec * sampling_rate
)

analysis_end_sample = min(
    int(analysis_duration_sec * sampling_rate),
    len(eda_clean)
)

window_features = []

for start_sample in range(
    ignore_samples,
    analysis_end_sample - window_samples + 1,
    window_samples
):
    end_sample = start_sample + window_samples

    start_sec = start_sample / sampling_rate
    end_sec = end_sample / sampling_rate

    window_clean = eda_clean[
        start_sample:end_sample
    ]

    window_scl = scl[
        start_sample:end_sample
    ]

    window_phasic = scr_phasic[
        start_sample:end_sample
    ]

    window_peaks = scr_peaks[
        start_sample:end_sample
    ]

    window_amplitude = scr_amplitude[
        start_sample:end_sample
    ]

    peak_mask = window_peaks == 1

    peak_amplitudes = window_amplitude[
        peak_mask
    ]

    peak_amplitudes = peak_amplitudes[
        np.isfinite(peak_amplitudes)
    ]

    # SCL 斜率
    x = np.arange(
        len(window_scl)
    ) / sampling_rate

    if len(window_scl) > 1:
        scl_slope = np.polyfit(
            x,
            window_scl,
            1
        )[0]
    else:
        scl_slope = 0.0

    if len(peak_amplitudes) > 0:
        scr_amplitude_mean = float(
            np.mean(peak_amplitudes)
        )

        scr_amplitude_max = float(
            np.max(peak_amplitudes)
        )
    else:
        scr_amplitude_mean = 0.0
        scr_amplitude_max = 0.0

    window_features.append({
        "start_sec": start_sec,
        "end_sec": end_sec,

        "eda_mean": float(
            np.mean(window_clean)
        ),

        "eda_std": float(
            np.std(window_clean)
        ),

        "scl_mean": float(
            np.mean(window_scl)
        ),

        "scl_std": float(
            np.std(window_scl)
        ),

        "scl_slope": float(
            scl_slope
        ),

        "scr_count": int(
            np.sum(peak_mask)
        ),

        "scr_amplitude_mean":
            scr_amplitude_mean,

        "scr_amplitude_max":
            scr_amplitude_max,

        "scr_phasic_max": float(
            np.max(window_phasic)
        )
    })


feature_df = pd.DataFrame(
    window_features
)

if feature_df.empty:
    raise ValueError(
        "前 60 秒資料不足，無法建立時間窗。"
    )


# ============================================================
# 11. 使用 10～30 秒建立暫時基線
# ============================================================

baseline_start_sec = 10
baseline_end_sec = 30

baseline_df = feature_df[
    (
        feature_df["start_sec"]
        >= baseline_start_sec
    )
    &
    (
        feature_df["end_sec"]
        <= baseline_end_sec
    )
].copy()

if len(baseline_df) < 2:
    raise ValueError(
        "基線時間窗不足。"
    )


# ============================================================
# 12. 計算門檻
# ============================================================

def calculate_upper_threshold(
    series,
    minimum_increase=0.0
):
    mean_value = float(
        series.mean()
    )

    std_value = float(
        series.std(ddof=0)
    )

    statistical_threshold = (
        mean_value + 2 * std_value
    )

    minimum_threshold = (
        mean_value + minimum_increase
    )

    return max(
        statistical_threshold,
        minimum_threshold
    )


scl_mean_threshold = calculate_upper_threshold(
    baseline_df["scl_mean"],
    minimum_increase=0.05
)

scl_slope_threshold = calculate_upper_threshold(
    baseline_df["scl_slope"],
    minimum_increase=0.002
)

scr_count_threshold = max(
    calculate_upper_threshold(
        baseline_df["scr_count"],
        minimum_increase=1
    ),
    2
)

scr_amplitude_threshold = max(
    calculate_upper_threshold(
        baseline_df["scr_amplitude_max"],
        minimum_increase=0.05
    ),
    0.05
)


# ============================================================
# 13. 判斷每個 10 秒時間窗
# ============================================================

anomaly_scores = []
anomaly_labels = []
anomaly_reasons = []

print("\n================ 判斷規則 ================")
print("符合 0～1 項：Normal")
print("符合 2 項：Suspected_Arousal")
print("符合 3～4 項：High_Arousal")
print("==========================================")

for _, row in feature_df.iterrows():

    score = 0
    reasons = []

    # 條件一：SCL 平均值偏高
    if row["scl_mean"] > scl_mean_threshold:
        score += 1
        reasons.append("SCL偏高")

    # 條件二：SCL 持續上升
    if row["scl_slope"] > scl_slope_threshold:
        score += 1
        reasons.append("SCL持續上升")

    # 條件三：SCR 峰值數量偏多
    if row["scr_count"] >= scr_count_threshold:
        score += 1
        reasons.append("SCR峰值偏多")

    # 條件四：SCR 振幅偏大
    if row["scr_amplitude_max"] >= scr_amplitude_threshold:
        score += 1
        reasons.append("SCR振幅偏大")

    # 根據符合條件數量判斷
    if score <= 1:
        label = "Normal"
    elif score == 2:
        label = "Suspected_Arousal"
    else:
        label = "High_Arousal"

    # 顯示每個時間窗的判斷結果
    print("-" * 60)
    print(
        f"時間窗：{row['start_sec']:.0f} ～ "
        f"{row['end_sec']:.0f} 秒"
    )
    print(f"符合條件數：{score}")
    print(f"判斷結果：{label}")

    if reasons:
        print("符合條件：")
        for reason in reasons:
            print(f"  ✓ {reason}")
    else:
        print("符合條件：無")

    print()

    # 每個時間窗都必須儲存一次結果
    anomaly_scores.append(score)
    anomaly_labels.append(label)

    if reasons:
        anomaly_reasons.append("、".join(reasons))
    else:
        anomaly_reasons.append("未明顯偏離基線")


# 迴圈完成後，再把五個時間窗的結果加入 DataFrame
feature_df["anomaly_score"] = anomaly_scores
feature_df["eda_label"] = anomaly_labels
feature_df["reason"] = anomaly_reasons


# ============================================================
# 14. 顯示前 60 秒判斷結果
# ============================================================

print("\n" + "=" * 60)
print("前 60 秒 EDA 判斷結果")
print("=" * 60)

print(
    feature_df[
        [
            "start_sec",
            "end_sec",
            "scl_mean",
            "scl_slope",
            "scr_count",
            "scr_amplitude_max",
            "anomaly_score",
            "eda_label",
            "reason"
        ]
    ].to_string(index=False)
)

print("\n判斷門檻：")
print("SCL 平均值門檻：", scl_mean_threshold)
print("SCL 斜率門檻：", scl_slope_threshold)
print("SCR 數量門檻：", scr_count_threshold)
print("SCR 振幅門檻：", scr_amplitude_threshold)

# ============================================================
# 15. 匯出完整 EDA 處理結果
# ============================================================

processed_df = pd.DataFrame({
    "time_sec": time,
    "eda_raw": eda_raw,
    "eda_clean": eda_clean,
    "scl_tonic": scl,
    "scr_phasic": scr_phasic,
    "scr_onset":
        signals["SCR_Onsets"].to_numpy(),
    "scr_peak": scr_peaks,
    "scr_amplitude": scr_amplitude
})


if "SCR_Height" in signals.columns:
    processed_df[
        "scr_height"
    ] = signals[
        "SCR_Height"
    ].to_numpy()


if "SCR_RiseTime" in signals.columns:
    processed_df[
        "scr_rise_time"
    ] = signals[
        "SCR_RiseTime"
    ].to_numpy()


if "SCR_Recovery" in signals.columns:
    processed_df[
        "scr_recovery"
    ] = signals[
        "SCR_Recovery"
    ].to_numpy()


if "SCR_RecoveryTime" in signals.columns:
    processed_df[
        "scr_recovery_time"
    ] = signals[
        "SCR_RecoveryTime"
    ].to_numpy()


processed_df.to_csv(
    processed_output_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. 匯出異常時間窗結果
# ============================================================

feature_df.to_csv(
    anomaly_output_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. 設定前 60 秒的繪圖範圍
# ============================================================

samples_to_plot = min(
    int(
        sampling_rate
        * plot_duration_sec
    ),
    len(eda_data)
)

plot_time = time[
    :samples_to_plot
]

peak_indices_plot = peak_indices[
    peak_indices < samples_to_plot
]


# ============================================================
# 18. 圖一：原始 EDA
# ============================================================

plt.figure(
    1,
    figsize=(12, 5)
)

plt.plot(
    plot_time,
    eda_raw[:samples_to_plot],
    label="Raw EDA"
)

plt.xlabel("Time (seconds)")
plt.ylabel("EDA")
plt.title(
    f"Raw EDA Signal - "
    f"First {plot_duration_sec} Seconds"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=False)


# ============================================================
# 19. 圖二：Clean EDA + 異常區段
# ============================================================

plt.figure(
    2,
    figsize=(12, 5)
)

plt.plot(
    plot_time,
    eda_clean[:samples_to_plot],
    label="Clean EDA"
)

# 標示基線區段
baseline_plot_end = min(
    baseline_end_sec,
    plot_duration_sec
)

if baseline_start_sec < plot_duration_sec:
    plt.axvspan(
        baseline_start_sec,
        baseline_plot_end,
        alpha=0.10,
        label="Baseline"
    )

# 標示疑似與高度喚醒區段
suspected_added = False
high_added = False

for _, row in feature_df.iterrows():

    if row["start_sec"] >= plot_duration_sec:
        continue

    region_end = min(
        row["end_sec"],
        plot_duration_sec
    )

    if (
        row["eda_label"]
        == "Suspected_Arousal"
    ):
        plt.axvspan(
            row["start_sec"],
            region_end,
            alpha=0.20,
            label=(
                "Suspected Arousal"
                if not suspected_added
                else None
            )
        )

        suspected_added = True

    elif (
        row["eda_label"]
        == "High_Arousal"
    ):
        plt.axvspan(
            row["start_sec"],
            region_end,
            alpha=0.35,
            label=(
                "High Arousal"
                if not high_added
                else None
            )
        )

        high_added = True


plt.xlabel("Time (seconds)")
plt.ylabel("EDA")
plt.title(
    f"Clean EDA and Arousal Detection - "
    f"First {plot_duration_sec} Seconds"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=False)


# ============================================================
# 20. 圖三：SCL（Tonic）
# ============================================================

plt.figure(
    3,
    figsize=(12, 5)
)

plt.plot(
    plot_time,
    scl[:samples_to_plot],
    label="SCL / Tonic"
)

plt.axhline(
    y=scl_mean_threshold,
    linestyle="--",
    label="SCL Threshold"
)

plt.xlabel("Time (seconds)")
plt.ylabel("SCL")
plt.title(
    f"SCL (Tonic Component) - "
    f"First {plot_duration_sec} Seconds"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=False)


# ============================================================
# 21. 圖四：SCR（Phasic）
# ============================================================

plt.figure(
    4,
    figsize=(12, 5)
)

plt.plot(
    plot_time,
    scr_phasic[:samples_to_plot],
    label="SCR / Phasic"
)

plt.xlabel("Time (seconds)")
plt.ylabel("SCR")
plt.title(
    f"SCR (Phasic Component) - "
    f"First {plot_duration_sec} Seconds"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=False)


# ============================================================
# 22. 圖五：SCR Peaks
# ============================================================

plt.figure(
    5,
    figsize=(12, 5)
)

plt.plot(
    plot_time,
    scr_phasic[:samples_to_plot],
    label="SCR / Phasic"
)

if len(peak_indices_plot) > 0:
    plt.scatter(
        time[peak_indices_plot],
        scr_phasic[peak_indices_plot],
        label="SCR Peaks"
    )

plt.xlabel("Time (seconds)")
plt.ylabel("SCR")
plt.title(
    f"SCR Peaks - "
    f"First {plot_duration_sec} Seconds"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=False)


# ============================================================
# 23. 顯示完整統計結果
# ============================================================

print("\n" + "=" * 60)
print("EDA 整體分析結果")
print("=" * 60)

print(
    "原始 EDA 平均值：",
    np.mean(eda_raw)
)

print(
    "原始 EDA 最大值：",
    np.max(eda_raw)
)

print(
    "原始 EDA 最小值：",
    np.min(eda_raw)
)

print(
    "原始 EDA 標準差：",
    np.std(eda_raw)
)

print(
    "\nSCL 平均值：",
    np.mean(scl)
)

print(
    "SCL 最大值：",
    np.max(scl)
)

print(
    "SCL 最小值：",
    np.min(scl)
)

print(
    "\nSCR 峰值總數：",
    len(peak_indices)
)


if len(peak_indices) > 0:

    valid_amplitudes = signals.loc[
        signals["SCR_Peaks"] == 1,
        "SCR_Amplitude"
    ].dropna()

    if len(valid_amplitudes) > 0:

        print(
            "SCR 平均振幅：",
            valid_amplitudes.mean()
        )

        print(
            "SCR 最大振幅：",
            valid_amplitudes.max()
        )


print("\n已輸出完整 EDA 資料：")
print(processed_output_path)

print("\n已輸出異常時間窗資料：")
print(anomaly_output_path)

print("=" * 60)


# 保持程式執行，讓五張圖同時存在
input("按 Enter 結束程式...")