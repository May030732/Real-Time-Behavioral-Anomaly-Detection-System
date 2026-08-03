import mne
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.ion()
# ============================================================
# 1. 設定檔案路徑
# ============================================================
file_path = Path(
    r"C:\Users\lin shan mei\Desktop\專題暑期進度\scientisst"
    r"\scientisst-move-annotated-wearable-multimodal-biosignals-recorded-during-everyday-life-activities-in-naturalistic-environments-1.0.1"
    r"\03FH\empatica.edf"
)

# CSV 會輸出到 EDF 檔案所在的資料夾
output_path = file_path.parent / "eda_processed_data.csv"

# 要顯示前幾秒
plot_duration_sec = 60


# ============================================================
# 2. 檢查檔案是否存在
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
# 4. 檢查並取得 EDA 通道
# ============================================================
eda_channel = "eda:dry"

if eda_channel not in raw.ch_names:
    raise ValueError(
        f"找不到通道：{eda_channel}\n"
        f"目前可使用的通道：{raw.ch_names}"
    )

eda_data = raw.get_data(picks=[eda_channel])[0]

sampling_rate = float(raw.info["sfreq"])
time = raw.times


print("EDA 通道：", eda_channel)
print("EDA 資料筆數：", len(eda_data))
print("EDA 取樣率：", sampling_rate, "Hz")
print("EDA 總長度：", len(eda_data) / sampling_rate, "秒")
print("EDA 前 10 筆：", eda_data[:10])


# ============================================================
# 5. 檢查無效值
# ============================================================
invalid_count = np.sum(~np.isfinite(eda_data))

if invalid_count > 0:
    print(f"發現 {invalid_count} 筆 NaN 或無限值，將使用插值處理。")

    eda_series = pd.Series(eda_data)

    eda_data = (
        eda_series
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(method="linear")
        .bfill()
        .ffill()
        .to_numpy()
    )


# ============================================================
# 6. 使用 NeuroKit2 處理 EDA
#
# EDA_Raw     ：原始 EDA
# EDA_Clean   ：清理後 EDA
# EDA_Tonic   ：SCL，慢速變化
# EDA_Phasic  ：SCR，快速變化
# SCR_Peaks   ：SCR 峰值位置
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
# 8. 設定繪圖範圍
# ============================================================
samples_to_plot = min(
    int(sampling_rate * plot_duration_sec),
    len(eda_data)
)

plot_time = time[:samples_to_plot]


# ============================================================
# 9. 圖一：原始 EDA
# ============================================================
plt.figure(figsize=(12, 5))

plt.plot(
    plot_time,
    signals["EDA_Raw"].iloc[:samples_to_plot],
    label="Raw EDA"
)

plt.xlabel("Time (seconds)")
plt.ylabel("EDA")
plt.title(f"Raw EDA Signal - First {plot_duration_sec} Seconds")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 10. 圖二：Clean EDA
# ============================================================
plt.figure(figsize=(12, 5))

plt.plot(
    plot_time,
    signals["EDA_Clean"].iloc[:samples_to_plot],
    label="Clean EDA"
)

plt.xlabel("Time (seconds)")
plt.ylabel("EDA")
plt.title(f"Clean EDA Signal - First {plot_duration_sec} Seconds")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 11. 圖三：SCL（Tonic）
# ============================================================
plt.figure(figsize=(12, 5))

plt.plot(
    plot_time,
    signals["EDA_Tonic"].iloc[:samples_to_plot],
    label="SCL / Tonic"
)

plt.xlabel("Time (seconds)")
plt.ylabel("SCL")
plt.title(
    f"SCL (Tonic Component) - First {plot_duration_sec} Seconds"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 12. 圖四：SCR（Phasic）
# ============================================================
plt.figure(figsize=(12, 5))

plt.plot(
    plot_time,
    signals["EDA_Phasic"].iloc[:samples_to_plot],
    label="SCR / Phasic"
)

plt.xlabel("Time (seconds)")
plt.ylabel("SCR")
plt.title(
    f"SCR (Phasic Component) - First {plot_duration_sec} Seconds"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 13. 找出 SCR Peaks
# ============================================================
peak_indices = np.where(
    signals["SCR_Peaks"].to_numpy() == 1
)[0]

# 只保留繪圖範圍內的峰值
peak_indices_plot = peak_indices[
    peak_indices < samples_to_plot
]


# ============================================================
# 14. 圖五：SCR 與峰值位置
# ============================================================
plt.figure(figsize=(12, 5))

plt.plot(
    plot_time,
    signals["EDA_Phasic"].iloc[:samples_to_plot],
    label="SCR / Phasic"
)

if len(peak_indices_plot) > 0:
    plt.scatter(
        time[peak_indices_plot],
        signals["EDA_Phasic"].iloc[
            peak_indices_plot
        ],
        label="SCR Peaks"
    )

plt.xlabel("Time (seconds)")
plt.ylabel("SCR")
plt.title(
    f"SCR Peaks - First {plot_duration_sec} Seconds"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 15. 建立輸出資料表
# ============================================================
df = pd.DataFrame({
    "time_sec": time,
    "eda_raw": signals["EDA_Raw"].to_numpy(),
    "eda_clean": signals["EDA_Clean"].to_numpy(),
    "scl_tonic": signals["EDA_Tonic"].to_numpy(),
    "scr_phasic": signals["EDA_Phasic"].to_numpy(),
    "scr_onset": signals["SCR_Onsets"].to_numpy(),
    "scr_peak": signals["SCR_Peaks"].to_numpy(),
    "scr_amplitude": signals["SCR_Amplitude"].to_numpy()
})

# 某些 NeuroKit2 版本會有 SCR_Recovery
if "SCR_Recovery" in signals.columns:
    df["scr_recovery"] = signals[
        "SCR_Recovery"
    ].to_numpy()


# ============================================================
# 16. 匯出 CSV
# ============================================================
df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. 顯示基本統計結果
# ============================================================
print("\n" + "=" * 60)
print("EDA 分析結果")
print("=" * 60)

print("原始 EDA 平均值：", signals["EDA_Raw"].mean())
print("原始 EDA 最大值：", signals["EDA_Raw"].max())
print("原始 EDA 最小值：", signals["EDA_Raw"].min())
print("原始 EDA 標準差：", signals["EDA_Raw"].std())

print("\nSCL 平均值：", signals["EDA_Tonic"].mean())
print("SCL 最大值：", signals["EDA_Tonic"].max())
print("SCL 最小值：", signals["EDA_Tonic"].min())

print("\nSCR 峰值總數：", len(peak_indices))

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

print("\n已輸出 CSV：")
print(output_path)

print("=" * 60)
input("按 Enter 結束...")