import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input
)

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# 1. Dataset 檔案
# ==========================================

FALL_DATASETS = [
    "datasets/fall1_dataset.npz",
    "datasets/fall2_dataset.npz",
    "datasets/fall4_dataset.npz",
    "datasets/fall6_dataset.npz",
    "datasets/fall8_dataset.npz",
    "datasets/fall9_dataset.npz",
    "datasets/fall10_dataset.npz",
    "datasets/fall11_dataset.npz"
]

NONFALL_DATASETS = [
    "datasets/nonfall_dataset.npz",
    "datasets/nonfall_turn_dataset.npz",
    "datasets/nonfall_back_dataset.npz",
    #"datasets/nonfall_lying_dataset.npz"
]


# ==========================================
# 2. 讀取所有 Fall Dataset
# ==========================================

fall_X_list = []
fall_y_list = []


print()
print("==============================")
print("讀取 Fall Dataset")
print("==============================")


for file_name in FALL_DATASETS:

    print()
    print("讀取：", file_name)

    data = np.load(
        file_name
    )

    X_temp = data["X"]
    y_temp = data["y"]


    # ======================================
    # 只留下 label = 1 的 Fall Sequence
    # ======================================

    fall_mask = (
        y_temp == 1
    )

    X_temp = X_temp[
        fall_mask
    ]

    y_temp = y_temp[
        fall_mask
    ]


    print(
        "Fall sequences：",
        len(X_temp)
    )


    # 有資料才加入
    if len(X_temp) > 0:

        fall_X_list.append(
            X_temp
        )

        fall_y_list.append(
            y_temp
        )


# ==========================================
# 3. 合併 Fall Dataset
# ==========================================

if len(fall_X_list) == 0:

    print(
        "錯誤：沒有找到任何 Fall Dataset"
    )

    exit()


X_fall = np.concatenate(
    fall_X_list,
    axis=0
)

y_fall = np.concatenate(
    fall_y_list,
    axis=0
)


print()
print("==============================")
print("全部 Fall")
print("==============================")

print(
    "X_fall Shape：",
    X_fall.shape
)

print(
    "y_fall Shape：",
    y_fall.shape
)

print(
    "Fall 總數：",
    len(y_fall)
)


# ==========================================
# 4. 讀取所有 Non-Fall Dataset
# ==========================================

nonfall_X_list = []
nonfall_y_list = []


print()
print("==============================")
print("讀取 Non-Fall Dataset")
print("==============================")


for file_name in NONFALL_DATASETS:

    print()
    print(
        "讀取：",
        file_name
    )


    data = np.load(
        file_name
    )


    X_temp = data["X"]
    y_temp = data["y"]


    # ======================================
    # 只留下 label = 0 的 Non-Fall
    # ======================================

    nonfall_mask = (
        y_temp == 0
    )


    X_temp = X_temp[
        nonfall_mask
    ]

    y_temp = y_temp[
        nonfall_mask
    ]


    print(
        "Non-Fall sequences：",
        len(X_temp)
    )


    if len(X_temp) > 0:

        nonfall_X_list.append(
            X_temp
        )

        nonfall_y_list.append(
            y_temp
        )


# ==========================================
# 5. 合併所有 Non-Fall
# ==========================================

if len(nonfall_X_list) == 0:

    print(
        "錯誤：沒有找到任何 Non-Fall Dataset"
    )

    exit()


X_nonfall = np.concatenate(
    nonfall_X_list,
    axis=0
)

y_nonfall = np.concatenate(
    nonfall_y_list,
    axis=0
)


print()
print("==============================")
print("全部 Non-Fall")
print("==============================")

print(
    "X_nonfall Shape：",
    X_nonfall.shape
)

print(
    "y_nonfall Shape：",
    y_nonfall.shape
)

print(
    "Non-Fall 總數：",
    len(y_nonfall)
)


# ==========================================
# 6. 合併 Fall + Non-Fall
# ==========================================

X = np.concatenate(
    [
        X_fall,
        X_nonfall
    ],
    axis=0
)


y = np.concatenate(
    [
        y_fall,
        y_nonfall
    ],
    axis=0
)


# ==========================================
# 7. 資料統計
# ==========================================

print()
print("==============================")
print("完整 Dataset")
print("==============================")

print(
    "X Shape：",
    X.shape
)

print(
    "y Shape：",
    y.shape
)

print(
    "Fall：",
    np.sum(y == 1)
)

print(
    "Non-Fall：",
    np.sum(y == 0)
)


# ==========================================
# 8. 檢查資料異常
# ==========================================

nan_count = np.isnan(
    X
).sum()


inf_count = np.isinf(
    X
).sum()


print()
print("==============================")
print("資料檢查")
print("==============================")

print(
    "NaN 數量：",
    nan_count
)

print(
    "Inf 數量：",
    inf_count
)


# 如果有 NaN / Inf
# 全部轉成 0
X = np.nan_to_num(
    X,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ==========================================
# 9. 打亂 Dataset
# ==========================================

random_indices = np.random.permutation(
    len(X)
)

X = X[
    random_indices
]

y = y[
    random_indices
]


# ==========================================
# 10. Train / Test Split
# ==========================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)


print()
print("==============================")
print("Train / Test Dataset")
print("==============================")

print(
    "X_train：",
    X_train.shape
)

print(
    "X_test：",
    X_test.shape
)

print(
    "Train Fall：",
    np.sum(y_train == 1)
)

print(
    "Train Non-Fall：",
    np.sum(y_train == 0)
)

print(
    "Test Fall：",
    np.sum(y_test == 1)
)

print(
    "Test Non-Fall：",
    np.sum(y_test == 0)
)


# ==========================================
# 11. Class Weight
# ==========================================
#
# 因為：
#
# Fall 資料比較少
# Non-Fall 資料比較多
#
# 所以讓 Fall 的 Loss 權重更大
#
# ==========================================

classes = np.unique(
    y_train
)


weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)


class_weight = {

    int(classes[i]):
    float(weights[i])

    for i in range(
        len(classes)
    )
}


print()
print("==============================")
print("Class Weight")
print("==============================")

print(
    class_weight
)


# ==========================================
# 12. 建立 LSTM
# ==========================================

model = Sequential([

    Input(
        shape=(
            X.shape[1],
            X.shape[2]
        )
    ),

    # --------------------------------------
    # 第一層 LSTM
    # --------------------------------------

    LSTM(
        64,
        return_sequences=True
    ),

    Dropout(
        0.3
    ),

    # --------------------------------------
    # 第二層 LSTM
    # --------------------------------------

    LSTM(
        32
    ),

    Dropout(
        0.3
    ),

    # --------------------------------------
    # Dense
    # --------------------------------------

    Dense(
        32,
        activation="relu"
    ),

    Dropout(
        0.2
    ),

    # --------------------------------------
    # Binary Classification
    #
    # 0 = Non_Fall
    # 1 = Fall
    # --------------------------------------

    Dense(
        1,
        activation="sigmoid"
    )
])


# ==========================================
# 13. Compile
# ==========================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[

        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


print()
print("==============================")
print("Model")
print("==============================")

model.summary()


# ==========================================
# 14. Early Stopping
# ==========================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True
)


# ==========================================
# 15. 訓練
# ==========================================

print()
print("==============================")
print("開始訓練")
print("==============================")


history = model.fit(

    X_train,
    y_train,

    # 20% training data
    # 再當 validation
    validation_split=0.2,

    epochs=100,

    batch_size=16,

    class_weight=class_weight,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ==========================================
# 16. Test
# ==========================================

print()
print("==============================")
print("Test Result")
print("==============================")


test_result = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print(
    "Loss：",
    test_result[0]
)

print(
    "Accuracy：",
    test_result[1]
)

print(
    "Precision：",
    test_result[2]
)

print(
    "Recall：",
    test_result[3]
)


# ==========================================
# 17. Test Prediction
# ==========================================

probabilities = model.predict(
    X_test,
    verbose=0
)


# ==========================================
# 18. Probability -> Label
# ==========================================
#
# >= 0.5
# Fall
#
# < 0.5
# Non-Fall
#
# ==========================================

y_pred = (

    probabilities
    .flatten()

    >= 0.5

).astype(
    int
)


# ==========================================
# 19. Confusion Matrix
# ==========================================

print()
print("==============================")
print("Confusion Matrix")
print("==============================")


cm = confusion_matrix(
    y_test,
    y_pred
)


print(
    cm
)


# ==========================================
# 20. Confusion Matrix 解釋
# ==========================================
#
# [[TN FP]
#  [FN TP]]
#
# TN：
# Non-Fall 判成 Non-Fall
#
# FP：
# Non-Fall 誤判成 Fall
#
# FN：
# Fall 漏判成 Non-Fall
#
# TP：
# Fall 正確判成 Fall
#
# ==========================================

if cm.shape == (2, 2):

    TN = cm[0][0]
    FP = cm[0][1]

    FN = cm[1][0]
    TP = cm[1][1]


    print()
    print(
        "True Non-Fall：",
        TN
    )

    print(
        "False Fall：",
        FP
    )

    print(
        "Missed Fall：",
        FN
    )

    print(
        "Correct Fall：",
        TP
    )


# ==========================================
# 21. Classification Report
# ==========================================

print()
print("==============================")
print("Classification Report")
print("==============================")


print(

    classification_report(

        y_test,

        y_pred,

        target_names=[
            "Non_Fall",
            "Fall"
        ],

        zero_division=0
    )
)


# ==========================================
# 22. 看一些 Probability
# ==========================================

print()
print("==============================")
print("Prediction Example")
print("==============================")


example_count = min(
    20,
    len(y_test)
)


for i in range(
    example_count
):

    true_label = (
        "Fall"
        if y_test[i] == 1
        else "Non_Fall"
    )

    predicted_label = (
        "Fall"
        if y_pred[i] == 1
        else "Non_Fall"
    )


    probability = float(
        probabilities[i][0]
    )


    print(

        f"{i:02d}"

        f" | True: {true_label:8}"

        f" | Predict: {predicted_label:8}"

        f" | Fall Probability: {probability:.3f}"
    )


# ==========================================
# 23. 儲存模型
# ==========================================

MODEL_FILE = (
    "fall_lstm_model.keras"
)


model.save(
    MODEL_FILE
)


print()
print("==============================")

print(
    "模型已儲存：",
    MODEL_FILE
)

print("==============================")