import csv
import math
import re
import statistics
from collections import Counter, deque
from pathlib import Path


# ============================================================
# 修改成你轉換後 CSV 所在的資料夾
# ============================================================
INPUT_FOLDER = Path(
    r"C:\Users\USER\Desktop\學校功課\專題"
    r"\capture24\capture24\Output_Capture24"
)

# 特徵檔輸出位置
OUTPUT_FILE = INPUT_FOLDER / "capture24_features_v1.csv"

# Capture-24 約為 100 Hz
SAMPLING_RATE = 100

# 2 秒視窗
WINDOW_SECONDS = 2
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS   # 200 筆

# 50% 重疊
STEP_SIZE = WINDOW_SIZE // 2                   # 100 筆

# 第一版先跑 10 人；確認成功後改成 None 跑全部
MAX_FILES = None

# 每個受試者每種標籤最多保留多少視窗
# 避免 sleeping 數量大到壓倒其他類別
MAX_WINDOWS_PER_CLASS_PER_SUBJECT = 3000


FEATURE_COLUMNS = [
    "participant",
    "label",

    "x_mean", "x_std", "x_min", "x_max",
    "x_range", "x_rms", "x_energy",

    "y_mean", "y_std", "y_min", "y_max",
    "y_range", "y_rms", "y_energy",

    "z_mean", "z_std", "z_min", "z_max",
    "z_range", "z_rms", "z_energy",

    "mag_mean", "mag_std", "mag_min", "mag_max",
    "mag_range", "mag_rms", "mag_energy",
]


def extract_met(activity_text: str) -> float | None:
    """
    從：
    '7030 sleeping;MET 0.95'
    擷取 0.95。
    """
    match = re.search(
        r"MET\s*([0-9]+(?:\.[0-9]+)?)",
        activity_text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def met_to_label(met: float) -> str:
    """將 MET 轉成四種活動強度。"""
    if met < 1.5:
        return "sedentary"

    if met < 3.0:
        return "light"

    if met < 6.0:
        return "moderate"

    return "vigorous"


def calculate_axis_features(values: list[float]) -> list[float]:
    """計算單一軸的基本時域特徵。"""
    count = len(values)

    mean_value = statistics.fmean(values)

    if count > 1:
        std_value = statistics.stdev(values)
    else:
        std_value = 0.0

    min_value = min(values)
    max_value = max(values)
    range_value = max_value - min_value

    squared_sum = sum(value * value for value in values)

    rms_value = math.sqrt(squared_sum / count)
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
    window_rows: list[tuple[float, float, float, str]],
) -> tuple[list[float], str] | None:
    """
    將一個 2 秒視窗轉成：
    特徵 + 多數標籤。
    """
    x_values = [row[0] for row in window_rows]
    y_values = [row[1] for row in window_rows]
    z_values = [row[2] for row in window_rows]
    labels = [row[3] for row in window_rows]

    # 視窗內使用出現次數最多的標籤
    majority_label, majority_count = Counter(labels).most_common(1)[0]

    # 若視窗內標籤太混亂，略過轉換邊界
    purity = majority_count / len(labels)

    if purity < 0.8:
        return None

    magnitude_values = [
        math.sqrt(x * x + y * y + z * z)
        for x, y, z in zip(x_values, y_values, z_values)
    ]

    features = []

    features.extend(calculate_axis_features(x_values))
    features.extend(calculate_axis_features(y_values))
    features.extend(calculate_axis_features(z_values))
    features.extend(calculate_axis_features(magnitude_values))

    return features, majority_label


def process_file(
    input_path: Path,
    writer: csv.writer,
) -> int:
    """逐列讀取一位受試者的 CSV，切窗並輸出特徵。"""
    participant = input_path.stem.replace(
        "_accelerometer",
        "",
    )

    print(f"\n處理：{participant}")

    class_counts: Counter[str] = Counter()
    window_buffer: deque[tuple[float, float, float, str]] = deque()

    output_count = 0
    rows_since_last_window = 0

    with input_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
        errors="replace",
    ) as input_file:

        reader = csv.DictReader(input_file)

        required_columns = {
            "acc_x",
            "acc_y",
            "acc_z",
            "activity",
        }

        if reader.fieldnames is None:
            print("  ⚠ 沒有欄位名稱，跳過。")
            return 0

        missing = required_columns - set(reader.fieldnames)

        if missing:
            print(f"  ⚠ 缺少欄位：{sorted(missing)}")
            return 0

        for row in reader:
            try:
                x = float(row["acc_x"])
                y = float(row["acc_y"])
                z = float(row["acc_z"])
            except (ValueError, TypeError):
                continue

            met = extract_met(row["activity"])

            if met is None:
                continue

            label = met_to_label(met)

            window_buffer.append((x, y, z, label))
            rows_since_last_window += 1

            if len(window_buffer) < WINDOW_SIZE:
                continue

            # 初次滿窗，或之後每隔 STEP_SIZE 筆建立一個視窗
            if (
                len(window_buffer) == WINDOW_SIZE
                and rows_since_last_window >= STEP_SIZE
            ):
                result = process_window(list(window_buffer))

                if result is not None:
                    features, majority_label = result

                    if (
                        class_counts[majority_label]
                        < MAX_WINDOWS_PER_CLASS_PER_SUBJECT
                    ):
                        writer.writerow(
                            [participant, majority_label]
                            + [f"{value:.8f}" for value in features]
                        )

                        class_counts[majority_label] += 1
                        output_count += 1

                # 保留後半窗，形成 50% overlap
                for _ in range(STEP_SIZE):
                    window_buffer.popleft()

                rows_since_last_window = 0

    print(f"  ✓ 輸出 {output_count:,} 個視窗")
    print(f"  類別數量：{dict(class_counts)}")

    return output_count


def main() -> None:
    csv_files = sorted(
        INPUT_FOLDER.glob("*_accelerometer.csv")
    )

    if MAX_FILES is not None:
        csv_files = csv_files[:MAX_FILES]

    if not csv_files:
        print("❌ 找不到 *_accelerometer.csv")
        print(INPUT_FOLDER)
        return

    print(f"找到 {len(csv_files)} 份資料")
    print(f"視窗大小：{WINDOW_SIZE} 筆")
    print(f"移動步長：{STEP_SIZE} 筆")
    print(f"特徵輸出：{OUTPUT_FILE}")

    total_windows = 0

    with OUTPUT_FILE.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as output_file:

        writer = csv.writer(output_file)
        writer.writerow(FEATURE_COLUMNS)

        for input_path in csv_files:
            total_windows += process_file(
                input_path,
                writer,
            )

    print("\n" + "=" * 60)
    print("特徵擷取完成")
    print(f"總視窗數：{total_windows:,}")
    print(f"輸出檔案：{OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()