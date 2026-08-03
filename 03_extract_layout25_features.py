import csv
import math
import statistics
from collections import Counter, deque
from pathlib import Path


# ============================================================
# 只需要修改這裡：
# 放置 P01_RightWrist.csv、P02_RightWrist.csv... 的資料夾
# ============================================================
INPUT_FOLDER = Path(
    r"C:\Users\USER\Desktop\學校功課\專題\dataset\dataset\Output_RightWrist"
)

# 特徵檔輸出位置
OUTPUT_FILE = (
    INPUT_FOLDER
    / "layout25_acc_gyr_features_v1.csv"
)


# ============================================================
# 資料集設定
# ============================================================

# 論文資料取樣率為 60 Hz
SAMPLING_RATE = 60

# 每個視窗 2 秒
WINDOW_SECONDS = 2

# 60 Hz × 2 秒 = 120 筆
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS

# 50% overlap
# 每次向前移動 60 筆
STEP_SIZE = WINDOW_SIZE // 2

# None 代表處理全部檔案
# 測試時可改成 1 或 3
MAX_FILES = None

# Activity = 0 代表 Other／無效活動，預設跳過
SKIP_ACTIVITY_ZERO = True


# ============================================================
# 原始 CSV 必須具備的欄位
# ============================================================
REQUIRED_COLUMNS = [
    "Activity",

    "Acc_X_RightWrist",
    "Acc_Y_RightWrist",
    "Acc_Z_RightWrist",

    "Gyr_X_RightWrist",
    "Gyr_Y_RightWrist",
    "Gyr_Z_RightWrist",
]


# 每個訊號計算 7 個時域特徵
FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "rms",
    "energy",
]


# 八組訊號：
# Acc X/Y/Z/Magnitude
# Gyr X/Y/Z/Magnitude
SIGNAL_NAMES = [
    "acc_x",
    "acc_y",
    "acc_z",
    "acc_mag",

    "gyr_x",
    "gyr_y",
    "gyr_z",
    "gyr_mag",
]


# 自動產生欄位名稱
FEATURE_COLUMNS = [
    "participant",
    "label",
]

for signal_name in SIGNAL_NAMES:
    for feature_name in FEATURE_NAMES:
        FEATURE_COLUMNS.append(
            f"{signal_name}_{feature_name}"
        )


def normalize_column_name(name: str) -> str:
    """移除欄位名稱前後空白及 BOM。"""
    return name.strip().lstrip("\ufeff")


def normalize_activity(value: str) -> str | None:
    """
    將 Activity 統一成字串形式。

    例如：
    7.0 → "7"
    10.0 → "10"
    """
    clean_value = value.strip()

    if not clean_value:
        return None

    try:
        numeric_value = float(clean_value)

        # 7.0 轉成 "7"
        if numeric_value.is_integer():
            return str(int(numeric_value))

        return str(numeric_value)

    except ValueError:
        # 若 Activity 本身是英文，仍可保留
        return clean_value


def calculate_signal_features(
    values: list[float],
) -> list[float]:
    """
    對單一訊號計算七個時域特徵：

    mean
    std
    min
    max
    range
    rms
    energy
    """
    count = len(values)

    if count == 0:
        raise ValueError("訊號資料為空。")

    mean_value = statistics.fmean(values)

    if count > 1:
        std_value = statistics.stdev(values)
    else:
        std_value = 0.0

    min_value = min(values)
    max_value = max(values)
    range_value = max_value - min_value

    squared_sum = sum(
        value * value
        for value in values
    )

    rms_value = math.sqrt(
        squared_sum / count
    )

    # 與第一版一致：
    # 使用平均平方值作為 energy
    energy_value = squared_sum / count

    return [
        mean_value,
        std_value,
        min_value,
        max_value,
        range_value,
        rms_value,
        energy_value,
    ]


def process_window(
    window_rows: list[
        tuple[
            float,
            float,
            float,
            float,
            float,
            float,
            str,
        ]
    ],
) -> tuple[list[float], str] | None:
    """
    將一個視窗轉換成：
    56個特徵 + 1個Activity標籤。
    """

    labels = [
        row[6]
        for row in window_rows
    ]

    unique_labels = set(labels)

    # 作者原始流程會略過含多種活動的視窗
    if len(unique_labels) != 1:
        return None

    label = labels[0]

    if SKIP_ACTIVITY_ZERO and label == "0":
        return None

    acc_x = [row[0] for row in window_rows]
    acc_y = [row[1] for row in window_rows]
    acc_z = [row[2] for row in window_rows]

    gyr_x = [row[3] for row in window_rows]
    gyr_y = [row[4] for row in window_rows]
    gyr_z = [row[5] for row in window_rows]

    acc_mag = [
        math.sqrt(
            x * x
            + y * y
            + z * z
        )
        for x, y, z in zip(
            acc_x,
            acc_y,
            acc_z,
        )
    ]

    gyr_mag = [
        math.sqrt(
            x * x
            + y * y
            + z * z
        )
        for x, y, z in zip(
            gyr_x,
            gyr_y,
            gyr_z,
        )
    ]

    features = []

    # 順序必須與 SIGNAL_NAMES 相同
    for signal in [
        acc_x,
        acc_y,
        acc_z,
        acc_mag,
        gyr_x,
        gyr_y,
        gyr_z,
        gyr_mag,
    ]:
        features.extend(
            calculate_signal_features(signal)
        )

    return features, label


def process_file(
    input_path: Path,
    writer: csv.writer,
) -> tuple[int, Counter]:
    """
    逐列讀取一位受試者資料，
    切成滑動視窗並輸出特徵。
    """
    participant = input_path.stem

    # 清理常見檔名後綴
    participant = participant.replace(
        "_RightWrist",
        "",
    )

    participant = participant.replace(
        "_rightwrist",
        "",
    )

    print(f"\n正在處理：{input_path.name}")
    print(f"受試者：{participant}")

    window_buffer: deque = deque(
        maxlen=WINDOW_SIZE
    )

    rows_since_last_window = 0
    output_count = 0
    skipped_mixed_windows = 0
    activity_counts: Counter[str] = Counter()

    with input_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
        errors="replace",
    ) as input_file:

        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            print("  ⚠ 找不到欄位名稱，已跳過。")
            return 0, activity_counts

        # 清理原始欄位名稱
        cleaned_fieldnames = [
            normalize_column_name(column)
            for column in reader.fieldnames
        ]

        reader.fieldnames = cleaned_fieldnames

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in cleaned_fieldnames
        ]

        if missing_columns:
            print("  ❌ 缺少必要欄位：")

            for column in missing_columns:
                print(f"    - {column}")

            print("  實際欄位：")

            for column in cleaned_fieldnames:
                print(f"    - {column}")

            return 0, activity_counts

        for row in reader:
            try:
                acc_x = float(
                    row["Acc_X_RightWrist"]
                )
                acc_y = float(
                    row["Acc_Y_RightWrist"]
                )
                acc_z = float(
                    row["Acc_Z_RightWrist"]
                )

                gyr_x = float(
                    row["Gyr_X_RightWrist"]
                )
                gyr_y = float(
                    row["Gyr_Y_RightWrist"]
                )
                gyr_z = float(
                    row["Gyr_Z_RightWrist"]
                )

            except (
                ValueError,
                TypeError,
                KeyError,
            ):
                continue

            activity = normalize_activity(
                row.get("Activity", "")
            )

            if activity is None:
                continue

            window_buffer.append(
                (
                    acc_x,
                    acc_y,
                    acc_z,
                    gyr_x,
                    gyr_y,
                    gyr_z,
                    activity,
                )
            )

            rows_since_last_window += 1

            if len(window_buffer) < WINDOW_SIZE:
                continue

            if rows_since_last_window < STEP_SIZE:
                continue

            result = process_window(
                list(window_buffer)
            )

            if result is None:
                skipped_mixed_windows += 1
            else:
                features, label = result

                writer.writerow(
                    [
                        participant,
                        label,
                    ]
                    + [
                        f"{value:.8f}"
                        for value in features
                    ]
                )

                output_count += 1
                activity_counts[label] += 1

            rows_since_last_window = 0

    print(
        f"  ✓ 輸出視窗：{output_count:,}"
    )

    print(
        "  略過跨活動或無效視窗："
        f"{skipped_mixed_windows:,}"
    )

    print(
        f"  Activity分布："
        f"{dict(activity_counts)}"
    )

    return output_count, activity_counts


def find_input_files() -> list[Path]:
    """
    尋找資料夾中的 CSV。

    會排除已產生的特徵檔。
    """
    csv_files = sorted(
        path
        for path in INPUT_FOLDER.glob("*.csv")
        if path.name
        != OUTPUT_FILE.name
    )

    if MAX_FILES is not None:
        csv_files = csv_files[:MAX_FILES]

    return csv_files


def main() -> None:
    if not INPUT_FOLDER.exists():
        print("❌ 找不到輸入資料夾：")
        print(INPUT_FOLDER)
        return

    csv_files = find_input_files()

    if not csv_files:
        print("❌ 找不到可處理的 CSV。")
        print(INPUT_FOLDER)
        return

    print("=" * 65)
    print("Layout25 RightWrist 特徵萃取")
    print(f"找到檔案：{len(csv_files)} 份")
    print(f"取樣率：{SAMPLING_RATE} Hz")
    print(
        f"視窗：{WINDOW_SECONDS} 秒 "
        f"（{WINDOW_SIZE} 筆）"
    )
    print(
        f"移動步長：{STEP_SIZE} 筆 "
        "（50% overlap）"
    )
    print(
        f"特徵數："
        f"{len(FEATURE_COLUMNS) - 2}"
    )
    print(f"輸出：{OUTPUT_FILE}")
    print("=" * 65)

    total_windows = 0
    total_activity_counts: Counter[str] = Counter()
    success_files = 0

    with OUTPUT_FILE.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as output_file:

        writer = csv.writer(output_file)
        writer.writerow(FEATURE_COLUMNS)

        for input_path in csv_files:
            output_count, activity_counts = (
                process_file(
                    input_path,
                    writer,
                )
            )

            if output_count > 0:
                success_files += 1

            total_windows += output_count
            total_activity_counts.update(
                activity_counts
            )

    print("\n" + "=" * 65)
    print("特徵萃取完成")
    print(
        f"成功產生資料的檔案："
        f"{success_files}/{len(csv_files)}"
    )
    print(
        f"總視窗數：{total_windows:,}"
    )
    print(
        f"總Activity分布："
        f"{dict(total_activity_counts)}"
    )
    print(f"輸出檔案：{OUTPUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()