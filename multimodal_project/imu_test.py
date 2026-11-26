import math
import statistics
from pathlib import Path

import joblib
import numpy as np


# ============================================================
# 修改處：改為相對路徑，指向 multimodal_project/weights/ 內的模型檔
# ============================================================
MODEL_FILE = Path("weights/layout25_intensity_random_forest_v1.joblib")

VALID_LABELS = {
    "calm",
    "slightly_intense",
    "fierce",
}

_MODEL_PACKAGE = None


def load_model():
    global _MODEL_PACKAGE

    if _MODEL_PACKAGE is not None:
        return _MODEL_PACKAGE

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "找不到 IMU 模型檔：\n"
            f"{MODEL_FILE}\n\n"
            "請確認 layout25_intensity_random_forest_v1.joblib 已放入 weights/ 資料夾。"
        )

    package = joblib.load(MODEL_FILE)

    if not isinstance(package, dict):
        raise ValueError("模型檔格式不正確：預期 joblib 內容為 dictionary。")

    required_keys = {"model", "feature_names"}
    missing_keys = required_keys - set(package.keys())
    if missing_keys:
        raise ValueError(f"模型檔缺少必要欄位：{sorted(missing_keys)}")

    _MODEL_PACKAGE = package
    return _MODEL_PACKAGE


def calculate_signal_features(values):
    values = [float(value) for value in values]
    count = len(values)

    if count == 0:
        raise ValueError("IMU 訊號不可為空。")

    if not all(math.isfinite(value) for value in values):
        raise ValueError("IMU 訊號包含 NaN 或 Infinity。")

    mean_value = statistics.fmean(values)
    std_value = statistics.stdev(values) if count > 1 else 0.0
    min_value = min(values)
    max_value = max(values)
    range_value = max_value - min_value

    squared_sum = sum(value * value for value in values)
    rms_value = math.sqrt(squared_sum / count)
    energy_value = squared_sum / count

    return {
        "mean": mean_value,
        "std": std_value,
        "min": min_value,
        "max": max_value,
        "range": range_value,
        "rms": rms_value,
        "energy": energy_value,
    }


def calculate_magnitude(x_values, y_values, z_values):
    return [
        math.sqrt((x * x) + (y * y) + (z * z))
        for x, y, z in zip(x_values, y_values, z_values)
    ]


def build_feature_dict(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z):
    signals = {
        "acc_x": [float(v) for v in acc_x],
        "acc_y": [float(v) for v in acc_y],
        "acc_z": [float(v) for v in acc_z],
        "gyr_x": [float(v) for v in gyr_x],
        "gyr_y": [float(v) for v in gyr_y],
        "gyr_z": [float(v) for v in gyr_z],
    }

    lengths = {len(values) for values in signals.values()}
    if len(lengths) != 1:
        raise ValueError("Acc/Gyr 六個軸的資料筆數必須完全相同。")

    window_length = next(iter(lengths))
    if window_length == 0:
        raise ValueError("IMU window 是空的。")

    package = load_model()
    expected_window_size = package.get("window_size")
    if expected_window_size is not None and window_length != int(expected_window_size):
        raise ValueError(
            f"IMU window 長度不符：收到 {window_length} 筆，預期 {expected_window_size} 筆。"
        )

    signals["acc_mag"] = calculate_magnitude(
        signals["acc_x"], signals["acc_y"], signals["acc_z"]
    )
    signals["gyr_mag"] = calculate_magnitude(
        signals["gyr_x"], signals["gyr_y"], signals["gyr_z"]
    )

    signal_order = [
        "acc_x", "acc_y", "acc_z", "acc_mag",
        "gyr_x", "gyr_y", "gyr_z", "gyr_mag",
    ]

    feature_dict = {}
    for signal_name in signal_order:
        result = calculate_signal_features(signals[signal_name])
        for feature_name, feature_value in result.items():
            feature_dict[f"{signal_name}_{feature_name}"] = feature_value

    if len(feature_dict) != 56:
        raise RuntimeError(f"特徵計算數量錯誤：目前得到 {len(feature_dict)} 個，預期應為 56 個。")

    return feature_dict


def predict_from_feature_dict(feature_dict, return_probabilities=False):
    package = load_model()
    model = package["model"]
    feature_names = package["feature_names"]

    missing_features = [f for f in feature_names if f not in feature_dict]
    if missing_features:
        raise ValueError(f"缺少模型需要的特徵：{missing_features}")

    feature_vector = [float(feature_dict[f]) for f in feature_names]
    x = np.asarray([feature_vector], dtype=np.float32)

    status = str(model.predict(x)[0])
    if status not in VALID_LABELS:
        raise ValueError(f"模型輸出了未知狀態：{status}")

    if not return_probabilities:
        return status

    probabilities = {}
    if hasattr(model, "predict_proba"):
        raw_probabilities = model.predict_proba(x)[0]
        probabilities = {
            str(label): float(prob)
            for label, prob in zip(model.classes_, raw_probabilities)
        }

    return status, probabilities


def predict_imu_window(
    acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, return_probabilities=False
):
    feature_dict = build_feature_dict(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
    return predict_from_feature_dict(
        feature_dict, return_probabilities=return_probabilities
    )


def predict_imu_rows(rows, return_probabilities=False):
    rows = list(rows)
    if not rows:
        raise ValueError("IMU rows 不可為空。")

    acc_x, acc_y, acc_z = [], [], []
    gyr_x, gyr_y, gyr_z = [], [], []

    for index, row in enumerate(rows, start=1):
        if len(row) < 6:
            raise ValueError(f"第 {index} 筆 IMU 資料不足 6 個值。")
        acc_x.append(float(row[0]))
        acc_y.append(float(row[1]))
        acc_z.append(float(row[2]))
        gyr_x.append(float(row[3]))
        gyr_y.append(float(row[4]))
        gyr_z.append(float(row[5]))

    return predict_imu_window(
        acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z,
        return_probabilities=return_probabilities,
    )


def get_model_info():
    package = load_model()
    return {
        "model_file": str(MODEL_FILE),
        "labels": package.get("labels", sorted(VALID_LABELS)),
        "sampling_rate": package.get("sampling_rate"),
        "window_seconds": package.get("window_seconds"),
        "window_size": package.get("window_size"),
        "sensor_location": package.get("sensor_location"),
        "number_of_features": len(package["feature_names"]),
    }
