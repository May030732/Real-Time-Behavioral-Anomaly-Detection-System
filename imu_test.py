"""
imu_test.py

用途：
1. 載入 06_train_layout25_intensity_random_forest.py 訓練出的模型。
2. 接收一個 RightWrist IMU 時間窗：
   Acc_X / Acc_Y / Acc_Z / Gyr_X / Gyr_Y / Gyr_Z
3. 用和 03_extract_layout25_features.py 相同的方法計算 56 個特徵。
4. 回傳三種狀態之一：
   - calm
   - slightly_intense
   - fierce

給 main_inference.py 使用的主要函式：
    status = imu_test.predict_imu_window(
        acc_x,
        acc_y,
        acc_z,
        gyr_x,
        gyr_y,
        gyr_z,
    )
"""

import math
import statistics
from pathlib import Path

import joblib
import numpy as np


# ============================================================
# 只需要確認這裡：
# 指向 06 訓練完成後產生的三分類模型
# ============================================================
MODEL_FILE = Path(
    r"C:\Users\USER\Desktop\學校功課\專題\dataset\dataset\Output_RightWrist\layout25_intensity_model_results\layout25_intensity_random_forest_v1.joblib"
)


# ============================================================
# 合法輸出狀態
# ============================================================
VALID_LABELS = {
    "calm",
    "slightly_intense",
    "fierce",
}


# ============================================================
# 模型快取
#
# 使用 lazy loading：
# import imu_test 時不會立刻讀模型，
# 第一次 predict 時才載入。
# ============================================================
_MODEL_PACKAGE = None


def load_model():
    """
    載入 06 訓練出的 joblib 模型。

    回傳 model_package。
    """

    global _MODEL_PACKAGE

    if _MODEL_PACKAGE is not None:
        return _MODEL_PACKAGE

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "找不到 IMU 模型檔：\n"
            f"{MODEL_FILE}\n\n"
            "請先執行 06_train_layout25_intensity_random_forest.py，"
            "或修改 imu_test.py 最上方的 MODEL_FILE。"
        )

    package = joblib.load(
        MODEL_FILE
    )

    if not isinstance(package, dict):
        raise ValueError(
            "模型檔格式不正確："
            "預期 joblib 內容為 dictionary。"
        )

    required_keys = {
        "model",
        "feature_names",
    }

    missing_keys = (
        required_keys
        - set(package.keys())
    )

    if missing_keys:
        raise ValueError(
            "模型檔缺少必要欄位："
            f"{sorted(missing_keys)}"
        )

    _MODEL_PACKAGE = package

    return _MODEL_PACKAGE


def calculate_signal_features(
    values,
):
    """
    對單一訊號計算與 03 相同的 7 個時域特徵：

    mean
    std
    min
    max
    range
    rms
    energy

    注意：
    std 使用 statistics.stdev，
    也就是 sample standard deviation。
    energy 使用平均平方值。
    """

    values = [
        float(value)
        for value in values
    ]

    count = len(values)

    if count == 0:
        raise ValueError(
            "IMU 訊號不可為空。"
        )

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "IMU 訊號包含 NaN 或 Infinity。"
        )

    mean_value = statistics.fmean(
        values
    )

    if count > 1:
        std_value = statistics.stdev(
            values
        )
    else:
        std_value = 0.0

    min_value = min(
        values
    )

    max_value = max(
        values
    )

    range_value = (
        max_value
        - min_value
    )

    squared_sum = sum(
        value * value
        for value in values
    )

    rms_value = math.sqrt(
        squared_sum / count
    )

    # 與 03 一致：
    # 平均平方值作為 energy
    energy_value = (
        squared_sum / count
    )

    return {
        "mean": mean_value,
        "std": std_value,
        "min": min_value,
        "max": max_value,
        "range": range_value,
        "rms": rms_value,
        "energy": energy_value,
    }


def calculate_magnitude(
    x_values,
    y_values,
    z_values,
):
    """
    計算三軸 magnitude：

    sqrt(x^2 + y^2 + z^2)
    """

    return [
        math.sqrt(
            (x * x)
            + (y * y)
            + (z * z)
        )
        for x, y, z in zip(
            x_values,
            y_values,
            z_values,
        )
    ]


def build_feature_dict(
    acc_x,
    acc_y,
    acc_z,
    gyr_x,
    gyr_y,
    gyr_z,
):
    """
    將一個 IMU window 轉換成 56 個特徵。

    8 組訊號：
        acc_x
        acc_y
        acc_z
        acc_mag
        gyr_x
        gyr_y
        gyr_z
        gyr_mag

    每組 7 個特徵：
        mean / std / min / max / range / rms / energy

    8 × 7 = 56 features
    """

    signals = {
        "acc_x": [
            float(v)
            for v in acc_x
        ],

        "acc_y": [
            float(v)
            for v in acc_y
        ],

        "acc_z": [
            float(v)
            for v in acc_z
        ],

        "gyr_x": [
            float(v)
            for v in gyr_x
        ],

        "gyr_y": [
            float(v)
            for v in gyr_y
        ],

        "gyr_z": [
            float(v)
            for v in gyr_z
        ],
    }

    lengths = {
        len(values)
        for values in signals.values()
    }

    if len(lengths) != 1:
        raise ValueError(
            "Acc/Gyr 六個軸的資料筆數必須完全相同。"
        )

    window_length = next(
        iter(lengths)
    )

    if window_length == 0:
        raise ValueError(
            "IMU window 是空的。"
        )

    # 讀模型資訊，確認訓練時的 window size
    package = load_model()

    expected_window_size = (
        package.get(
            "window_size"
        )
    )

    if (
        expected_window_size is not None
        and window_length
        != int(expected_window_size)
    ):
        raise ValueError(
            "IMU window 長度與模型訓練設定不同。\n"
            f"目前收到：{window_length} 筆\n"
            f"模型預期：{expected_window_size} 筆\n"
            "目前 Layout25 設定通常為 60 Hz × 2 秒 = 120 筆。"
        )

    # magnitude
    signals["acc_mag"] = (
        calculate_magnitude(
            signals["acc_x"],
            signals["acc_y"],
            signals["acc_z"],
        )
    )

    signals["gyr_mag"] = (
        calculate_magnitude(
            signals["gyr_x"],
            signals["gyr_y"],
            signals["gyr_z"],
        )
    )

    signal_order = [
        "acc_x",
        "acc_y",
        "acc_z",
        "acc_mag",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "gyr_mag",
    ]

    feature_dict = {}

    for signal_name in signal_order:

        result = (
            calculate_signal_features(
                signals[signal_name]
            )
        )

        for (
            feature_name,
            feature_value,
        ) in result.items():

            full_name = (
                f"{signal_name}_"
                f"{feature_name}"
            )

            feature_dict[
                full_name
            ] = feature_value

    if len(feature_dict) != 56:
        raise RuntimeError(
            "特徵計算數量錯誤："
            f"目前得到 {len(feature_dict)} 個，"
            "預期應為 56 個。"
        )

    return feature_dict


def predict_from_feature_dict(
    feature_dict,
    return_probabilities=False,
):
    """
    已經有 56 個特徵時，可以直接用這個函式預測。

    預設回傳：
        "calm"
        "slightly_intense"
        "fierce"

    若 return_probabilities=True，
    則回傳：
        (
            status,
            {
                "calm": ...,
                "slightly_intense": ...,
                "fierce": ...
            }
        )
    """

    package = load_model()

    model = package[
        "model"
    ]

    feature_names = package[
        "feature_names"
    ]

    missing_features = [
        feature_name
        for feature_name in feature_names
        if feature_name
        not in feature_dict
    ]

    if missing_features:
        raise ValueError(
            "缺少模型需要的特徵："
            f"{missing_features}"
        )

    # 必須完全按照訓練時 feature_names 的順序
    feature_vector = [
        float(
            feature_dict[
                feature_name
            ]
        )
        for feature_name
        in feature_names
    ]

    x = np.asarray(
        [feature_vector],
        dtype=np.float32,
    )

    prediction = (
        model.predict(x)[0]
    )

    status = str(
        prediction
    )

    if status not in VALID_LABELS:
        raise ValueError(
            "模型輸出了未知狀態："
            f"{status}"
        )

    if not return_probabilities:
        return status

    probabilities = {}

    if hasattr(
        model,
        "predict_proba",
    ):

        raw_probabilities = (
            model.predict_proba(x)[0]
        )

        probabilities = {
            str(label): float(probability)
            for label, probability
            in zip(
                model.classes_,
                raw_probabilities,
            )
        }

    return (
        status,
        probabilities,
    )


def predict_imu_window(
    acc_x,
    acc_y,
    acc_z,
    gyr_x,
    gyr_y,
    gyr_z,
    return_probabilities=False,
):
    """
    ★ 給 main_inference.py 使用的主要函式 ★

    輸入：
        一個完整 IMU window 的六軸資料。

    目前模型設定通常是：
        60 Hz
        2 秒
        = 每軸 120 筆

    輸出：
        calm
        slightly_intense
        fierce

    範例：

        status = imu_test.predict_imu_window(
            acc_x,
            acc_y,
            acc_z,
            gyr_x,
            gyr_y,
            gyr_z,
        )
    """

    feature_dict = (
        build_feature_dict(
            acc_x,
            acc_y,
            acc_z,
            gyr_x,
            gyr_y,
            gyr_z,
        )
    )

    return (
        predict_from_feature_dict(
            feature_dict,
            return_probabilities=(
                return_probabilities
            ),
        )
    )


def predict_imu_rows(
    rows,
    return_probabilities=False,
):
    """
    如果主程式拿到的是：

        [
            [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z],
            [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z],
            ...
        ]

    可以直接呼叫此函式。

    例如：
        status = imu_test.predict_imu_rows(rows)
    """

    rows = list(
        rows
    )

    if not rows:
        raise ValueError(
            "IMU rows 不可為空。"
        )

    acc_x = []
    acc_y = []
    acc_z = []

    gyr_x = []
    gyr_y = []
    gyr_z = []

    for index, row in enumerate(
        rows,
        start=1,
    ):

        if len(row) < 6:
            raise ValueError(
                f"第 {index} 筆 IMU 資料不足 6 個值。"
            )

        try:
            ax = float(row[0])
            ay = float(row[1])
            az = float(row[2])

            gx = float(row[3])
            gy = float(row[4])
            gz = float(row[5])

        except (
            ValueError,
            TypeError,
        ) as error:

            raise ValueError(
                f"第 {index} 筆 IMU 資料格式錯誤。"
            ) from error

        acc_x.append(ax)
        acc_y.append(ay)
        acc_z.append(az)

        gyr_x.append(gx)
        gyr_y.append(gy)
        gyr_z.append(gz)

    return (
        predict_imu_window(
            acc_x,
            acc_y,
            acc_z,
            gyr_x,
            gyr_y,
            gyr_z,
            return_probabilities=(
                return_probabilities
            ),
        )
    )


def get_model_info():
    """
    提供主程式或除錯時查看模型設定。
    """

    package = load_model()

    return {
        "model_file":
            str(MODEL_FILE),

        "labels":
            package.get(
                "labels",
                sorted(
                    VALID_LABELS
                ),
            ),

        "sampling_rate":
            package.get(
                "sampling_rate"
            ),

        "window_seconds":
            package.get(
                "window_seconds"
            ),

        "window_size":
            package.get(
                "window_size"
            ),

        "sensor_location":
            package.get(
                "sensor_location"
            ),

        "number_of_features":
            len(
                package[
                    "feature_names"
                ]
            ),
    }


# ============================================================
# 單獨執行 imu_test.py 時
#
# 不會亂造 IMU 資料去預測，
# 只檢查模型是否能正常載入。
# ============================================================
if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "IMU Test / Inference Module"
    )

    print(
        "=" * 60
    )

    try:

        info = get_model_info()

        print(
            "✓ 模型載入成功"
        )

        print(
            f"模型：{info['model_file']}"
        )

        print(
            f"輸出標籤：{info['labels']}"
        )

        print(
            f"取樣率：{info['sampling_rate']} Hz"
        )

        print(
            f"視窗秒數：{info['window_seconds']} 秒"
        )

        print(
            f"每個視窗：{info['window_size']} 筆"
        )

        print(
            f"特徵數：{info['number_of_features']}"
        )

        print(
            f"感測位置：{info['sensor_location']}"
        )

        print(
            "\n此檔案主要供 main_inference.py import。"
        )

        print(
            "\n使用方式："
        )

        print(
            "import imu_test"
        )

        print(
            "status = imu_test.predict_imu_window("
        )

        print(
            "    acc_x, acc_y, acc_z,"
        )

        print(
            "    gyr_x, gyr_y, gyr_z"
        )

        print(
            ")"
        )

        print(
            "\nstatus 會是："
        )

        print(
            "calm / slightly_intense / fierce"
        )

    except Exception as error:

        print(
            "❌ IMU 模組檢查失敗："
        )

        print(
            error
        )
