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
)
from sklearn.model_selection import GroupShuffleSplit


FEATURE_FILE = Path(
    r"C:\Users\USER\Desktop\學校功課\專題"
    r"\capture24\capture24\Output_Capture24"
    r"\capture24_features_v1.csv"
)

MODEL_FILE = FEATURE_FILE.parent / "random_forest_v1.joblib"
REPORT_FILE = FEATURE_FILE.parent / "random_forest_v1_report.txt"


def load_feature_file(
    feature_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """讀取特徵 CSV。"""
    x_rows = []
    labels = []
    groups = []

    with feature_file.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("特徵檔沒有欄位名稱。")

        feature_names = [
            column
            for column in reader.fieldnames
            if column not in {"participant", "label"}
        ]

        for row in reader:
            try:
                features = [
                    float(row[column])
                    for column in feature_names
                ]
            except (ValueError, TypeError):
                continue

            x_rows.append(features)
            labels.append(row["label"])
            groups.append(row["participant"])

    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(labels),
        np.asarray(groups),
        feature_names,
    )


def main() -> None:
    if not FEATURE_FILE.exists():
        print("❌ 找不到特徵檔：")
        print(FEATURE_FILE)
        return

    print("正在讀取特徵……")

    x, y, groups, feature_names = load_feature_file(
        FEATURE_FILE
    )

    print(f"視窗數：{len(y):,}")
    print(f"特徵數：{len(feature_names)}")
    print(f"受試者數：{len(set(groups))}")
    print(f"類別分布：{dict(Counter(y))}")

    # 依照「受試者」切分，而不是打散所有視窗
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(x, y, groups=groups)
    )

    x_train = x[train_indices]
    x_test = x[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    train_people = sorted(set(groups[train_indices]))
    test_people = sorted(set(groups[test_indices]))

    print("\n訓練受試者：", train_people)
    print("測試受試者：", test_people)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    print("\n開始訓練 Random Forest……")
    model.fit(x_train, y_train)

    print("訓練完成，開始測試……")
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0,
    )

    labels = sorted(set(y))

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print("\nAccuracy：", round(accuracy, 4))
    print(
        "Balanced accuracy：",
        round(balanced_accuracy, 4),
    )

    print("\nClassification report：")
    print(report)

    print("Confusion matrix：")
    print("標籤順序：", labels)
    print(matrix)

    # 顯示重要特徵
    importances = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    top_features = importances[:10]

    print("\n最重要的 10 個特徵：")

    for name, importance in top_features:
        print(f"{name}: {importance:.6f}")

    # 儲存模型與必要資訊
    model_package = {
        "model": model,
        "feature_names": feature_names,
        "labels": labels,
        "window_size": 200,
        "sampling_rate": 100,
    }

    joblib.dump(
        model_package,
        MODEL_FILE,
    )

    with REPORT_FILE.open(
        mode="w",
        encoding="utf-8",
    ) as report_file:

        report_file.write(
            f"Accuracy: {accuracy:.6f}\n"
        )

        report_file.write(
            "Balanced accuracy: "
            f"{balanced_accuracy:.6f}\n\n"
        )

        report_file.write(
            f"Train participants: {train_people}\n"
        )

        report_file.write(
            f"Test participants: {test_people}\n\n"
        )

        report_file.write(
            "Classification report:\n"
        )
        report_file.write(report)

        report_file.write(
            "\nConfusion matrix labels:\n"
        )
        report_file.write(str(labels))

        report_file.write(
            "\n\nConfusion matrix:\n"
        )
        report_file.write(str(matrix))

        report_file.write(
            "\n\nTop features:\n"
        )

        for name, importance in top_features:
            report_file.write(
                f"{name}: {importance:.8f}\n"
            )

    print("\n✓ 模型已儲存：")
    print(MODEL_FILE)

    print("\n✓ 評估報告已儲存：")
    print(REPORT_FILE)


if __name__ == "__main__":
    main()