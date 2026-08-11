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
    / "layout25_intensity_model_results"
)

MODEL_FILE = (
    OUTPUT_FOLDER
    / "layout25_intensity_random_forest_v1.joblib"
)

REPORT_FILE = (
    OUTPUT_FOLDER
    / "layout25_intensity_random_forest_v1_report.txt"
)

CONFUSION_MATRIX_FILE = (
    OUTPUT_FOLDER
    / "layout25_intensity_confusion_matrix_v1.csv"
)

FEATURE_IMPORTANCE_FILE = (
    OUTPUT_FOLDER
    / "layout25_intensity_feature_importance_v1.csv"
)


# ============================================================
# 原始 Activity ID 對照表
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
# 新的三分類標籤
#
# calm
#   1 Lying
#   2 Sitting
#   3 Standing
#   4 Slow walking
#   5 Moderate walking
#
# slightly_intense
#   6 Brisk walking
#   7 Ascending stairs
#   8 Descending stairs
#   9 Cycling
#
# fierce
#   10 Running
#   11 Jumping
#   12 Rowing
# ============================================================
INTENSITY_MAP = {
    1: "calm",
    2: "calm",
    3: "calm",
    4: "calm",
    5: "calm",

    6: "slightly_intense",
    7: "slightly_intense",
    8: "slightly_intense",
    9: "slightly_intense",

    10: "fierce",
    11: "fierce",
    12: "fierce",
}


# 固定顯示順序
INTENSITY_LABELS = [
    "calm",
    "slightly_intense",
    "fierce",
]


# ============================================================
# Random Forest 參數
# ============================================================
N_ESTIMATORS = 300
MAX_DEPTH = 25
MIN_SAMPLES_LEAF = 2

TEST_SIZE = 0.20
RANDOM_STATE = 42


def convert_activity_to_intensity(
    activity_label: int,
) -> str:
    """
    將原始 Activity 1～12
    轉換成三種活動強度標籤。
    """

    if activity_label not in INTENSITY_MAP:
        raise ValueError(
            f"未知的 Activity label：{activity_label}"
        )

    return INTENSITY_MAP[
        activity_label
    ]


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

    X：
        56個 IMU 特徵

    y：
        calm /
        slightly_intense /
        fierce

    groups：
        participant

    feature_names：
        特徵名稱
    """

    feature_rows = []
    labels = []
    participants = []

    invalid_rows = 0

    with feature_file.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
        errors="replace",
    ) as file:

        reader = csv.DictReader(
            file
        )

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

        # 自動取得全部 56 個特徵
        feature_names = [
            column
            for column
            in reader.fieldnames
            if column not in {
                "participant",
                "label",
            }
        ]

        if not feature_names:
            raise ValueError(
                "特徵檔中沒有模型特徵。"
            )

        for row in reader:

            try:
                participant = (
                    row["participant"]
                    .strip()
                )

                original_label = int(
                    float(
                        row["label"]
                        .strip()
                    )
                )

                # ================================
                # 關鍵：
                # 原本 1～12
                # 在這裡直接轉成三類
                # ================================
                intensity_label = (
                    convert_activity_to_intensity(
                        original_label
                    )
                )

                features = [
                    float(
                        row[column]
                    )
                    for column
                    in feature_names
                ]

                if not participant:
                    invalid_rows += 1
                    continue

                if not np.all(
                    np.isfinite(
                        features
                    )
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

            feature_rows.append(
                features
            )

            labels.append(
                intensity_label
            )

            participants.append(
                participant
            )

    if not feature_rows:
        raise ValueError(
            "沒有成功讀取任何有效資料。"
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
            dtype=str,
        ),

        np.asarray(
            participants,
            dtype=str,
        ),

        feature_names,
    )


def save_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
) -> None:
    """
    儲存 3×3 Confusion Matrix。
    """

    with CONFUSION_MATRIX_FILE.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            ["actual\\predicted"]
            + labels
        )

        for label, row in zip(
            labels,
            matrix,
        ):
            writer.writerow(
                [label]
                + row.tolist()
            )


def save_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
) -> list[
    tuple[str, float]
]:
    """
    儲存所有 56 個特徵重要度。
    """

    sorted_importances = sorted(
        zip(
            feature_names,
            importances,
        ),
        key=lambda item:
        item[1],
        reverse=True,
    )

    with FEATURE_IMPORTANCE_FILE.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

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


def print_distribution(
    title: str,
    labels: np.ndarray,
) -> None:
    """
    顯示三類數量。
    """

    counter = Counter(
        labels.tolist()
    )

    print(
        f"\n{title}"
    )

    for label in INTENSITY_LABELS:

        print(
            f"  {label}: "
            f"{counter.get(label, 0):,}"
        )


def write_distribution(
    file,
    title: str,
    labels: np.ndarray,
) -> None:
    """
    將三類分布寫進 report。
    """

    counter = Counter(
        labels.tolist()
    )

    file.write(
        f"{title}:\n"
    )

    for label in INTENSITY_LABELS:

        file.write(
            f"  {label}: "
            f"{counter.get(label, 0)}\n"
        )

    file.write(
        "\n"
    )


def main() -> None:

    # ========================================================
    # 1. 檢查輸入檔
    # ========================================================
    if not FEATURE_FILE.exists():

        print(
            "❌ 找不到特徵檔："
        )

        print(
            FEATURE_FILE
        )

        return

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 70
    )

    print(
        "Layout25 RightWrist"
    )

    print(
        "3-Class Intensity "
        "Random Forest"
    )

    print(
        "=" * 70
    )

    print(
        f"特徵檔："
        f"{FEATURE_FILE}"
    )

    print(
        f"輸出資料夾："
        f"{OUTPUT_FOLDER}"
    )

    # ========================================================
    # 2. 讀取資料
    # ========================================================
    print(
        "\n正在讀取特徵資料……"
    )

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

        print(
            f"❌ 讀取資料失敗："
            f"{error}"
        )

        return

    participant_names = sorted(
        set(
            groups.tolist()
        )
    )

    print(
        f"\n總視窗數："
        f"{len(y):,}"
    )

    print(
        f"特徵數："
        f"{len(feature_names)}"
    )

    print(
        f"受試者數："
        f"{len(participant_names)}"
    )

    print_distribution(
        "整體三分類分布：",
        y,
    )

    # ========================================================
    # 3. 依受試者切 Train / Test
    #
    # 保持與 04 一樣的 GroupShuffleSplit
    # ========================================================
    splitter = (
        GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    (
        train_indices,
        test_indices,
    ) = next(
        splitter.split(
            x,
            y,
            groups=groups,
        )
    )

    x_train = (
        x[train_indices]
    )

    x_test = (
        x[test_indices]
    )

    y_train = (
        y[train_indices]
    )

    y_test = (
        y[test_indices]
    )

    train_groups = (
        groups[train_indices]
    )

    test_groups = (
        groups[test_indices]
    )

    train_people = sorted(
        set(
            train_groups.tolist()
        )
    )

    test_people = sorted(
        set(
            test_groups.tolist()
        )
    )

    print(
        "\n訓練受試者："
    )

    print(
        train_people
    )

    print(
        "\n測試受試者："
    )

    print(
        test_people
    )

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
        &
        set(test_people)
    )

    if overlap_people:

        raise RuntimeError(
            "資料切分錯誤："
            "Train/Test 有相同受試者。"
        )

    print_distribution(
        "訓練集分布：",
        y_train,
    )

    print_distribution(
        "測試集分布：",
        y_test,
    )

    # ========================================================
    # 4. Random Forest
    # ========================================================
    model = (
        RandomForestClassifier(
            n_estimators=(
                N_ESTIMATORS
            ),

            max_depth=(
                MAX_DEPTH
            ),

            min_samples_leaf=(
                MIN_SAMPLES_LEAF
            ),

            # 三類數量不同
            # 自動調整權重
            class_weight="balanced",

            n_jobs=-1,

            random_state=(
                RANDOM_STATE
            ),

            verbose=0,
        )
    )

    print(
        "\n開始訓練 "
        "3-Class Random Forest……"
    )

    print(
        f"決策樹數量："
        f"{N_ESTIMATORS}"
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

        print(
            error
        )

        return

    print(
        "✓ 訓練完成"
    )

    # ========================================================
    # 5. 預測
    # ========================================================
    print(
        "\n開始測試模型……"
    )

    predictions = (
        model.predict(
            x_test
        )
    )

    # ========================================================
    # 6. 評估
    # ========================================================
    accuracy = (
        accuracy_score(
            y_test,
            predictions,
        )
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions,
        )
    )

    macro_f1 = (
        f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    weighted_f1 = (
        f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    report = (
        classification_report(
            y_test,
            predictions,

            labels=(
                INTENSITY_LABELS
            ),

            target_names=(
                INTENSITY_LABELS
            ),

            digits=4,

            zero_division=0,
        )
    )

    matrix = (
        confusion_matrix(
            y_test,
            predictions,

            labels=(
                INTENSITY_LABELS
            ),
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "三分類模型評估結果"
    )

    print(
        "=" * 70
    )

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

    print(
        "\nClassification Report："
    )

    print(
        report
    )

    print(
        "Confusion Matrix："
    )

    print(
        "順序："
    )

    print(
        INTENSITY_LABELS
    )

    print(
        matrix
    )

    # ========================================================
    # 7. Feature Importance
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

    print(
        "\n最重要的 15 個特徵："
    )

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
    # 8. 儲存 Confusion Matrix
    # ========================================================
    save_confusion_matrix(
        matrix,
        INTENSITY_LABELS,
    )

    # ========================================================
    # 9. 儲存模型
    # ========================================================
    model_package = {

        "model":
            model,

        "feature_names":
            feature_names,

        "labels":
            INTENSITY_LABELS,

        "intensity_map":
            INTENSITY_MAP,

        "original_activity_names":
            ACTIVITY_NAMES,

        "sampling_rate":
            60,

        "window_seconds":
            2,

        "window_size":
            120,

        "step_size":
            60,

        "overlap":
            0.5,

        "sensor_location":
            "RightWrist",

        "sensor_channels": [
            "Acc_X_RightWrist",
            "Acc_Y_RightWrist",
            "Acc_Z_RightWrist",

            "Gyr_X_RightWrist",
            "Gyr_Y_RightWrist",
            "Gyr_Z_RightWrist",
        ],

        "train_participants":
            train_people,

        "test_participants":
            test_people,

        "random_forest_parameters": {
            "n_estimators":
                N_ESTIMATORS,

            "max_depth":
                MAX_DEPTH,

            "min_samples_leaf":
                MIN_SAMPLES_LEAF,

            "class_weight":
                "balanced",

            "random_state":
                RANDOM_STATE,
        },

        "metrics": {

            "accuracy":
                float(
                    accuracy
                ),

            "balanced_accuracy":
                float(
                    balanced_accuracy
                ),

            "macro_f1":
                float(
                    macro_f1
                ),

            "weighted_f1":
                float(
                    weighted_f1
                ),
        },
    }

    joblib.dump(
        model_package,
        MODEL_FILE,
    )

    # ========================================================
    # 10. 儲存文字報告
    # ========================================================
    with REPORT_FILE.open(
        mode="w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Layout25 RightWrist "
            "3-Class Intensity "
            "Random Forest v1\n"
        )

        file.write(
            "=" * 70
        )

        file.write(
            "\n\n"
        )

        file.write(
            "Intensity Definition:\n"
        )

        file.write(
            "\ncalm:\n"
        )

        file.write(
            "  1 Lying\n"
            "  2 Sitting\n"
            "  3 Standing\n"
            "  4 Slow walking\n"
            "  5 Moderate walking\n"
        )

        file.write(
            "\nslightly_intense:\n"
        )

        file.write(
            "  6 Brisk walking\n"
            "  7 Ascending stairs\n"
            "  8 Descending stairs\n"
            "  9 Cycling\n"
        )

        file.write(
            "\nfierce:\n"
        )

        file.write(
            "  10 Running\n"
            "  11 Jumping\n"
            "  12 Rowing\n\n"
        )

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

        write_distribution(
            file,
            "Overall intensity distribution",
            y,
        )

        write_distribution(
            file,
            "Training intensity distribution",
            y_train,
        )

        write_distribution(
            file,
            "Testing intensity distribution",
            y_test,
        )

        file.write(
            f"Accuracy: "
            f"{accuracy:.6f}\n"
        )

        file.write(
            "Balanced Accuracy: "
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

        file.write(
            report
        )

        file.write(
            "\nConfusion Matrix Labels:\n"
        )

        file.write(
            str(
                INTENSITY_LABELS
            )
        )

        file.write(
            "\n\nConfusion Matrix:\n"
        )

        file.write(
            str(
                matrix
            )
        )

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
    # 11. 完成
    # ========================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "全部完成 🎉"
    )

    print(
        "=" * 70
    )

    print(
        "\n模型已儲存："
    )

    print(
        MODEL_FILE
    )

    print(
        "\n評估報告已儲存："
    )

    print(
        REPORT_FILE
    )

    print(
        "\nConfusion Matrix CSV："
    )

    print(
        CONFUSION_MATRIX_FILE
    )

    print(
        "\nFeature Importance CSV："
    )

    print(
        FEATURE_IMPORTANCE_FILE
    )


if __name__ == "__main__":
    main()