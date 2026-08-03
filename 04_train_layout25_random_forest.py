import csv
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# 03_extract_layout25_features.py 產生的特徵檔
# ============================================================
FEATURE_FILE = Path(
    r"C:\Users\USER\Desktop\學校功課\專題"
    r"\dataset\dataset\Output_RightWrist"
    r"\layout25_acc_gyr_features_v1.csv"
)


# ============================================================
# 輸出位置
# ============================================================
OUTPUT_FOLDER = (
    FEATURE_FILE.parent
    / "layout25_model_results"
)

MODEL_FILE = (
    OUTPUT_FOLDER
    / "layout25_acc_gyr_random_forest_v1.joblib"
)

REPORT_FILE = (
    OUTPUT_FOLDER
    / "layout25_acc_gyr_random_forest_v1_report.txt"
)

CONFUSION_MATRIX_FILE = (
    OUTPUT_FOLDER
    / "layout25_acc_gyr_confusion_matrix_v1.csv"
)

FEATURE_IMPORTANCE_FILE = (
    OUTPUT_FOLDER
    / "layout25_acc_gyr_feature_importance_v1.csv"
)


# ============================================================
# Activity ID 對照表
#
# 注意：key 全部使用 int，不再使用字串。
# ============================================================
ACTIVITY_NAMES = {
    1: "Lying",
    2: "Sitting",
    3: "Standing",
    4: "Slow walking",
    5: "Moderate walking",
    6: "Brisk walking",
    7: "Ascending stairs",
    8: "Descending stairs",
    9: "Cycling",
    10: "Running",
    11: "Jumping",
    12: "Rowing",
}


# ============================================================
# Random Forest 參數
# ============================================================
N_ESTIMATORS = 300
MAX_DEPTH = 25
MIN_SAMPLES_LEAF = 2

TEST_SIZE = 0.20
RANDOM_STATE = 42


def activity_display_name(label: int) -> str:
    """
    將 Activity ID 轉成顯示名稱。

    例如：
    3 → "3 - Standing"
    """
    activity_name = ACTIVITY_NAMES.get(
        int(label),
        "Unknown",
    )

    return f"{int(label)} - {activity_name}"


def load_feature_file(
    feature_file: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """
    讀取 03 輸出的特徵 CSV。

    回傳：
    X             模型輸入特徵
    y             Activity標籤（整數）
    groups        受試者編號
    feature_names 特徵名稱
    """
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    participants: list[str] = []

    invalid_rows = 0

    with feature_file.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
        errors="replace",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "特徵檔沒有欄位名稱。"
            )

        required_columns = {
            "participant",
            "label",
        }

        missing_columns = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "特徵檔缺少必要欄位："
                f"{sorted(missing_columns)}"
            )

        # 自動取得 participant、label 以外的全部特徵
        feature_names = [
            column
            for column in reader.fieldnames
            if column not in {
                "participant",
                "label",
            }
        ]

        if not feature_names:
            raise ValueError(
                "特徵檔中沒有可供模型使用的特徵。"
            )

        for row in reader:
            try:
                participant = (
                    row["participant"].strip()
                )

                # 例如 "3" 或 "3.0" 都統一轉成 int 3
                label = int(
                    float(
                        row["label"].strip()
                    )
                )

                features = [
                    float(row[column])
                    for column in feature_names
                ]

                if not participant:
                    invalid_rows += 1
                    continue

                # 只接受 Activity 1～12
                if label not in ACTIVITY_NAMES:
                    invalid_rows += 1
                    continue

                # 排除 NaN 與正負無限大
                if not np.all(
                    np.isfinite(features)
                ):
                    invalid_rows += 1
                    continue

            except (
                ValueError,
                TypeError,
                KeyError,
            ):
                invalid_rows += 1
                continue

            feature_rows.append(features)
            labels.append(label)
            participants.append(participant)

    if not feature_rows:
        raise ValueError(
            "沒有成功讀取任何有效特徵資料。"
        )

    print(
        f"略過無效資料列："
        f"{invalid_rows:,}"
    )

    return (
        np.asarray(
            feature_rows,
            dtype=np.float32,
        ),
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        np.asarray(
            participants,
            dtype=str,
        ),
        feature_names,
    )


def save_confusion_matrix(
    matrix: np.ndarray,
    labels: list[int],
) -> None:
    """把混淆矩陣存成 CSV。"""
    display_labels = [
        activity_display_name(label)
        for label in labels
    ]

    with CONFUSION_MATRIX_FILE.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["實際\\預測"]
            + display_labels
        )

        for display_label, row in zip(
            display_labels,
            matrix,
        ):
            writer.writerow(
                [display_label]
                + row.tolist()
            )


def save_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
) -> list[tuple[str, float]]:
    """排序並儲存所有特徵重要度。"""
    sorted_importances = sorted(
        zip(
            feature_names,
            importances,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    with FEATURE_IMPORTANCE_FILE.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "rank",
                "feature",
                "importance",
            ]
        )

        for rank, (
            feature_name,
            importance,
        ) in enumerate(
            sorted_importances,
            start=1,
        ):
            writer.writerow(
                [
                    rank,
                    feature_name,
                    f"{importance:.10f}",
                ]
            )

    return sorted_importances


def write_label_distribution(
    file,
    title: str,
    labels: np.ndarray,
) -> None:
    """把類別分布寫入文字報告。"""
    counts = Counter(
        int(label)
        for label in labels
    )

    file.write(f"{title}:\n")

    for label in sorted(counts):
        file.write(
            f"  {activity_display_name(label)}: "
            f"{counts[label]}\n"
        )

    file.write("\n")


def main() -> None:
    # ========================================================
    # 1. 檢查輸入檔
    # ========================================================
    if not FEATURE_FILE.exists():
        print("❌ 找不到特徵檔：")
        print(FEATURE_FILE)
        return

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("Layout25 RightWrist Random Forest")
    print(f"特徵檔：{FEATURE_FILE}")
    print(f"輸出資料夾：{OUTPUT_FOLDER}")
    print("=" * 70)

    # ========================================================
    # 2. 讀取特徵
    # ========================================================
    print("\n正在讀取特徵資料……")

    try:
        (
            x,
            y,
            groups,
            feature_names,
        ) = load_feature_file(
            FEATURE_FILE
        )

    except (
        ValueError,
        OSError,
    ) as error:
        print(f"❌ 讀取特徵失敗：{error}")
        return

    participant_names = sorted(
        set(groups.tolist())
    )

    # y 已經全部是整數，所以可以直接排序
    label_names = sorted(
        int(label)
        for label in set(y.tolist())
    )

    print(f"\n總視窗數：{len(y):,}")
    print(f"特徵數：{len(feature_names)}")
    print(
        f"受試者數："
        f"{len(participant_names)}"
    )

    print("\n整體 Activity 分布：")

    total_label_counts = Counter(
        int(label)
        for label in y
    )

    for label in label_names:
        print(
            f"  {activity_display_name(label)}："
            f"{total_label_counts[label]:,}"
        )

    # ========================================================
    # 3. 依受試者分訓練集與測試集
    # ========================================================
    if len(participant_names) < 2:
        print(
            "❌ 受試者不足，"
            "至少需要兩位才能切分資料。"
        )
        return

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(
            x,
            y,
            groups=groups,
        )
    )

    x_train = x[train_indices]
    x_test = x[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    train_groups = groups[train_indices]
    test_groups = groups[test_indices]

    train_people = sorted(
        set(train_groups.tolist())
    )

    test_people = sorted(
        set(test_groups.tolist())
    )

    print("\n訓練受試者：")
    print(train_people)

    print("\n測試受試者：")
    print(test_people)

    print(
        f"\n訓練視窗數："
        f"{len(y_train):,}"
    )

    print(
        f"測試視窗數："
        f"{len(y_test):,}"
    )

    overlap_people = (
        set(train_people)
        & set(test_people)
    )

    if overlap_people:
        raise RuntimeError(
            "資料切分錯誤："
            "有受試者同時出現在訓練集與測試集。"
        )

    # ========================================================
    # 4. 檢查類別是否完整
    # ========================================================
    train_labels = set(
        int(label)
        for label in y_train
    )

    test_labels = set(
        int(label)
        for label in y_test
    )

    missing_train_labels = (
        test_labels
        - train_labels
    )

    if missing_train_labels:
        print(
            "\n⚠ 測試集出現訓練集沒有的類別："
        )

        for label in sorted(
            missing_train_labels
        ):
            print(
                f"  - "
                f"{activity_display_name(label)}"
            )

        print(
            "模型沒有學過這些類別，"
            "相關結果需特別注意。"
        )

    print("\n訓練集 Activity 分布：")

    train_distribution = Counter(
        int(label)
        for label in y_train
    )

    for label in label_names:
        print(
            f"  {activity_display_name(label)}："
            f"{train_distribution.get(label, 0):,}"
        )

    print("\n測試集 Activity 分布：")

    test_distribution = Counter(
        int(label)
        for label in y_test
    )

    for label in label_names:
        print(
            f"  {activity_display_name(label)}："
            f"{test_distribution.get(label, 0):,}"
        )

    # ========================================================
    # 5. 建立 Random Forest
    # ========================================================
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,

        # 自動依照訓練資料的類別數量調整權重
        class_weight="balanced",

        # 使用所有可用CPU核心
        n_jobs=-1,

        random_state=RANDOM_STATE,
        verbose=0,
    )

    print("\n開始訓練 Random Forest……")
    print(
        f"決策樹數量：{N_ESTIMATORS}"
    )

    try:
        model.fit(
            x_train,
            y_train,
        )

    except Exception as error:
        print(
            "\n❌ 模型訓練失敗："
        )
        print(error)
        return

    print("✓ 訓練完成")

    # ========================================================
    # 6. 測試模型
    # ========================================================
    print("\n開始測試模型……")

    predictions = model.predict(
        x_test
    )

    # ========================================================
    # 7. 計算評估結果
    # ========================================================
    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions,
        )
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    display_names = [
        activity_display_name(label)
        for label in label_names
    ]

    report = classification_report(
        y_test,
        predictions,
        labels=label_names,
        target_names=display_names,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=label_names,
    )

    print("\n" + "=" * 70)
    print("模型評估結果")
    print("=" * 70)

    print(
        f"Accuracy："
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy："
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1："
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1："
        f"{weighted_f1:.4f}"
    )

    print("\nClassification Report：")
    print(report)

    print("Confusion Matrix：")
    print("標籤順序：")
    print(display_names)
    print(matrix)

    # ========================================================
    # 8. 特徵重要度
    # ========================================================
    sorted_importances = (
        save_feature_importance(
            feature_names,
            model.feature_importances_,
        )
    )

    top_features = (
        sorted_importances[:15]
    )

    print("\n最重要的 15 個特徵：")

    for rank, (
        feature_name,
        importance,
    ) in enumerate(
        top_features,
        start=1,
    ):
        print(
            f"{rank:02d}. "
            f"{feature_name}: "
            f"{importance:.8f}"
        )

    # ========================================================
    # 9. 儲存混淆矩陣
    # ========================================================
    save_confusion_matrix(
        matrix,
        label_names,
    )

    # ========================================================
    # 10. 儲存模型
    # ========================================================
    model_package = {
        "model": model,

        # 預測時必須維持相同特徵順序
        "feature_names": feature_names,

        "labels": label_names,
        "activity_names": ACTIVITY_NAMES,

        "sampling_rate": 60,
        "window_seconds": 2,
        "window_size": 120,
        "step_size": 60,
        "overlap": 0.5,

        "sensor_location": "RightWrist",

        "sensor_channels": [
            "Acc_X_RightWrist",
            "Acc_Y_RightWrist",
            "Acc_Z_RightWrist",
            "Gyr_X_RightWrist",
            "Gyr_Y_RightWrist",
            "Gyr_Z_RightWrist",
        ],

        "train_participants": train_people,
        "test_participants": test_people,

        "random_forest_parameters": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": (
                MIN_SAMPLES_LEAF
            ),
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },

        "metrics": {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(
                balanced_accuracy
            ),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(
                weighted_f1
            ),
        },
    }

    joblib.dump(
        model_package,
        MODEL_FILE,
    )

    # ========================================================
    # 11. 儲存文字報告
    # ========================================================
    with REPORT_FILE.open(
        mode="w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Layout25 RightWrist "
            "Random Forest v1\n"
        )

        file.write("=" * 70)
        file.write("\n\n")

        file.write(
            f"Feature file: "
            f"{FEATURE_FILE}\n"
        )

        file.write(
            f"Number of windows: "
            f"{len(y)}\n"
        )

        file.write(
            f"Number of features: "
            f"{len(feature_names)}\n"
        )

        file.write(
            f"Number of participants: "
            f"{len(participant_names)}\n\n"
        )

        file.write(
            f"Train participants: "
            f"{train_people}\n"
        )

        file.write(
            f"Test participants: "
            f"{test_people}\n\n"
        )

        file.write(
            f"Training windows: "
            f"{len(y_train)}\n"
        )

        file.write(
            f"Testing windows: "
            f"{len(y_test)}\n\n"
        )

        write_label_distribution(
            file,
            "Overall activity distribution",
            y,
        )

        write_label_distribution(
            file,
            "Training activity distribution",
            y_train,
        )

        write_label_distribution(
            file,
            "Testing activity distribution",
            y_test,
        )

        file.write(
            f"Accuracy: "
            f"{accuracy:.6f}\n"
        )

        file.write(
            f"Balanced Accuracy: "
            f"{balanced_accuracy:.6f}\n"
        )

        file.write(
            f"Macro F1: "
            f"{macro_f1:.6f}\n"
        )

        file.write(
            f"Weighted F1: "
            f"{weighted_f1:.6f}\n\n"
        )

        file.write(
            "Classification Report:\n"
        )

        file.write(report)

        file.write(
            "\nConfusion Matrix Labels:\n"
        )

        file.write(
            str(display_names)
        )

        file.write(
            "\n\nConfusion Matrix:\n"
        )

        file.write(str(matrix))

        file.write(
            "\n\nTop 15 Features:\n"
        )

        for rank, (
            feature_name,
            importance,
        ) in enumerate(
            top_features,
            start=1,
        ):
            file.write(
                f"{rank:02d}. "
                f"{feature_name}: "
                f"{importance:.10f}\n"
            )

    # ========================================================
    # 12. 完成
    # ========================================================
    print("\n" + "=" * 70)
    print("全部完成 🎉")
    print("=" * 70)

    print("\n模型已儲存：")
    print(MODEL_FILE)

    print("\n評估報告已儲存：")
    print(REPORT_FILE)

    print("\n混淆矩陣 CSV：")
    print(CONFUSION_MATRIX_FILE)

    print("\n特徵重要度 CSV：")
    print(FEATURE_IMPORTANCE_FILE)


if __name__ == "__main__":
    main()