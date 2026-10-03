import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dropout,
    Dense,
    Concatenate
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)


# =========================================================
# 1. 路徑設定
# =========================================================

PROJECT_ROOT = Path(
    r"C:\Users\USER\Desktop\學校功課\專題\101_vedio_pose_to_csv"
)

DATA_FOLDER = (
    PROJECT_ROOT
    / "10118_LSTM訓練資料集(V6_真人偏差修正版)"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "10119_LSTM模型(第六版_真人偏差修正)"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = (
    OUTPUT_FOLDER
    / "lstm_v6_dual_branch.keras"
)

HISTORY_PATH = (
    OUTPUT_FOLDER
    / "training_history_v6.csv"
)

REPORT_PATH = (
    OUTPUT_FOLDER
    / "test_report_v6.txt"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_FOLDER
    / "confusion_matrix_v6.png"
)

ACCURACY_CURVE_PATH = (
    OUTPUT_FOLDER
    / "accuracy_curve_v6.png"
)

LOSS_CURVE_PATH = (
    OUTPUT_FOLDER
    / "loss_curve_v6.png"
)


# =========================================================
# 2. 訓練參數
# =========================================================

SEQUENCE_LENGTH = 30

POSE_FEATURES = 26
MOTION_FEATURES = 26

EPOCHS = 50
BATCH_SIZE = 64

VIOLENCE_THRESHOLD = 0.5

RANDOM_SEED = 42


np.random.seed(
    RANDOM_SEED
)


# =========================================================
# 3. 載入 V6 Dataset
# =========================================================

print("=" * 75)
print("載入 V6 真人偏差修正版 Dataset")
print("=" * 75)


X_train = np.load(
    DATA_FOLDER
    / "X_train_v6_pose_motion.npy"
)

y_train = np.load(
    DATA_FOLDER
    / "y_train_v6.npy"
)


X_val = np.load(
    DATA_FOLDER
    / "X_val_v6_pose_motion.npy"
)

y_val = np.load(
    DATA_FOLDER
    / "y_val_v6.npy"
)


X_test = np.load(
    DATA_FOLDER
    / "X_test_v6_pose_motion.npy"
)

y_test = np.load(
    DATA_FOLDER
    / "y_test_v6.npy"
)


print()
print(
    "Train:",
    X_train.shape,
    y_train.shape
)

print(
    "Validation:",
    X_val.shape,
    y_val.shape
)

print(
    "Test:",
    X_test.shape,
    y_test.shape
)


# =========================================================
# 4. Dataset 檢查
# =========================================================

print()
print("=" * 75)
print("Dataset 檢查")
print("=" * 75)


for name, X, y in [

    (
        "Train",
        X_train,
        y_train
    ),

    (
        "Validation",
        X_val,
        y_val
    ),

    (
        "Test",
        X_test,
        y_test
    )
]:

    print()
    print(name)

    print(
        "  X shape:",
        X.shape
    )

    print(
        "  y shape:",
        y.shape
    )


    unique, counts = np.unique(
        y,
        return_counts=True
    )


    for label, count in zip(
        unique,
        counts
    ):

        print(
            f"  Label {label}: {count}"
        )


    nan_count = np.isnan(
        X
    ).sum()

    inf_count = np.isinf(
        X
    ).sum()


    print(
        f"  NaN: {nan_count}"
    )

    print(
        f"  Inf: {inf_count}"
    )


    if (
        nan_count > 0
        or inf_count > 0
    ):

        raise ValueError(
            f"{name} 發現 NaN / Inf"
        )


# =========================================================
# 5. Shape 安全檢查
# =========================================================

EXPECTED_SHAPE = (
    SEQUENCE_LENGTH,
    POSE_FEATURES + MOTION_FEATURES
)


if X_train.shape[1:] != EXPECTED_SHAPE:

    raise ValueError(
        f"Train shape 錯誤："
        f"{X_train.shape}"
    )


if X_val.shape[1:] != EXPECTED_SHAPE:

    raise ValueError(
        f"Validation shape 錯誤："
        f"{X_val.shape}"
    )


if X_test.shape[1:] != EXPECTED_SHAPE:

    raise ValueError(
        f"Test shape 錯誤："
        f"{X_test.shape}"
    )


# =========================================================
# 6. Pose / Motion 分開
# =========================================================
#
# V3/V6 feature 定義：
#
# [:, :, 0:26]  = Pose
# [:, :, 26:52] = Motion
#
# =========================================================

X_train_pose = X_train[
    :,
    :,
    :26
]

X_train_motion = X_train[
    :,
    :,
    26:52
]


X_val_pose = X_val[
    :,
    :,
    :26
]

X_val_motion = X_val[
    :,
    :,
    26:52
]


X_test_pose = X_test[
    :,
    :,
    :26
]

X_test_motion = X_test[
    :,
    :,
    26:52
]


print()
print("=" * 75)
print("Dual-Branch Input")
print("=" * 75)

print(
    "Train Pose:",
    X_train_pose.shape
)

print(
    "Train Motion:",
    X_train_motion.shape
)

print(
    "Validation Pose:",
    X_val_pose.shape
)

print(
    "Validation Motion:",
    X_val_motion.shape
)

print(
    "Test Pose:",
    X_test_pose.shape
)

print(
    "Test Motion:",
    X_test_motion.shape
)


# =========================================================
# 7. 建立 V6 Dual-Branch LSTM
# =========================================================
#
# 注意：
#
# V6 不載入 V5 model。
# 從頭建立相同 architecture。
#
# 目的：
# V5 vs V6 唯一主要實驗變因 = Training Data
#
# =========================================================

print()
print("=" * 75)
print("建立 V6 Dual-Branch LSTM")
print("=" * 75)


# ---------------------------------------------------------
# Pose Branch
# ---------------------------------------------------------

pose_input = Input(
    shape=(
        SEQUENCE_LENGTH,
        POSE_FEATURES
    ),
    name="pose_input"
)


pose_branch = LSTM(
    64,
    name="pose_lstm"
)(
    pose_input
)


pose_branch = Dropout(
    0.3,
    name="pose_dropout"
)(
    pose_branch
)


# ---------------------------------------------------------
# Motion Branch
# ---------------------------------------------------------

motion_input = Input(
    shape=(
        SEQUENCE_LENGTH,
        MOTION_FEATURES
    ),
    name="motion_input"
)


motion_branch = LSTM(
    64,
    name="motion_lstm"
)(
    motion_input
)


motion_branch = Dropout(
    0.3,
    name="motion_dropout"
)(
    motion_branch
)


# ---------------------------------------------------------
# Fusion
# ---------------------------------------------------------

merged = Concatenate(
    name="pose_motion_fusion"
)(
    [
        pose_branch,
        motion_branch
    ]
)


dense = Dense(
    32,
    activation="relu",
    name="fusion_dense"
)(
    merged
)


dense = Dropout(
    0.3,
    name="fusion_dropout"
)(
    dense
)


output = Dense(
    1,
    activation="sigmoid",
    name="violence_output"
)(
    dense
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = Model(
    inputs=[
        pose_input,
        motion_input
    ],
    outputs=output,
    name="V6_Dual_Branch_LSTM"
)


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy"
    ]
)


model.summary()


# =========================================================
# 8. Callbacks
# =========================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


model_checkpoint = ModelCheckpoint(

    filepath=MODEL_PATH,

    monitor="val_loss",

    save_best_only=True,

    verbose=1
)


# =========================================================
# 9. 訓練
# =========================================================

print()
print("=" * 75)
print("開始訓練 V6")
print("=" * 75)


history = model.fit(

    [
        X_train_pose,
        X_train_motion
    ],

    y_train,

    validation_data=(

        [
            X_val_pose,
            X_val_motion
        ],

        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        model_checkpoint
    ],

    verbose=1
)


# =========================================================
# 10. 儲存 Training History
# =========================================================

history_df = pd.DataFrame(
    history.history
)


history_df.to_csv(
    HISTORY_PATH,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 11. Test Evaluation
# =========================================================

print()
print("=" * 75)
print("V6 Test Evaluation")
print("=" * 75)


test_loss, test_accuracy = model.evaluate(

    [
        X_test_pose,
        X_test_motion
    ],

    y_test,

    verbose=0
)


print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)


# =========================================================
# 12. Test Prediction
# =========================================================

y_probability = model.predict(

    [
        X_test_pose,
        X_test_motion
    ],

    verbose=0
).reshape(-1)


y_pred = (
    y_probability
    >= VIOLENCE_THRESHOLD
).astype(
    np.int64
)


# =========================================================
# 13. Metrics
# =========================================================

violence_precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

violence_recall = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

violence_f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)


classification_text = classification_report(

    y_test,
    y_pred,

    target_names=[
        "Non-Violence",
        "Violence"
    ],

    digits=4,

    zero_division=0
)


cm = confusion_matrix(
    y_test,
    y_pred
)


print()
print(
    f"Violence Precision: "
    f"{violence_precision:.4f}"
)

print(
    f"Violence Recall: "
    f"{violence_recall:.4f}"
)

print(
    f"Violence F1: "
    f"{violence_f1:.4f}"
)


print()
print(
    classification_text
)


print(
    "Confusion Matrix:"
)

print(
    cm
)


# =========================================================
# 14. 儲存 Test Report
# =========================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "LSTM V6 Test Report\n"
    )

    file.write(
        "==============================\n\n"
    )


    file.write(
        "Architecture: Dual-Branch LSTM\n"
    )

    file.write(
        "Pose Input: (30, 26)\n"
    )

    file.write(
        "Motion Input: (30, 26)\n"
    )

    file.write(
        "Pose Branch: LSTM(64)\n"
    )

    file.write(
        "Motion Branch: LSTM(64)\n"
    )

    file.write(
        "Fusion: Concatenate -> Dense(32) -> Sigmoid\n\n"
    )


    file.write(
        "V6 Change:\n"
    )

    file.write(
        "V5 architecture retained; "
        "balanced real-world hard examples "
        "added to training data only.\n"
    )

    file.write(
        "Validation and Test sets remain unchanged.\n\n"
    )


    file.write(
        f"Train Samples: {len(X_train)}\n"
    )

    file.write(
        f"Validation Samples: {len(X_val)}\n"
    )

    file.write(
        f"Test Samples: {len(X_test)}\n\n"
    )


    file.write(
        f"Test Loss: "
        f"{test_loss:.4f}\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}\n"
    )

    file.write(
        f"Violence Precision: "
        f"{violence_precision:.4f}\n"
    )

    file.write(
        f"Violence Recall: "
        f"{violence_recall:.4f}\n"
    )

    file.write(
        f"Violence F1: "
        f"{violence_f1:.4f}\n\n"
    )


    file.write(
        "Classification Report\n"
    )

    file.write(
        "------------------------------\n"
    )

    file.write(
        classification_text
    )


    file.write(
        "\nConfusion Matrix\n"
    )

    file.write(
        "------------------------------\n"
    )

    file.write(
        str(cm)
    )

    file.write(
        "\n"
    )


# =========================================================
# 15. Confusion Matrix 圖
# =========================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "V6 Confusion Matrix"
)

plt.colorbar()


tick_marks = np.arange(
    2
)


plt.xticks(
    tick_marks,
    [
        "Non-Violence",
        "Violence"
    ]
)

plt.yticks(
    tick_marks,
    [
        "Non-Violence",
        "Violence"
    ]
)


threshold = (
    cm.max()
    / 2.0
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            str(
                cm[i, j]
            ),
            horizontalalignment="center",
            verticalalignment="center"
        )


plt.ylabel(
    "True Label"
)

plt.xlabel(
    "Predicted Label"
)

plt.tight_layout()


plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 16. Accuracy Curve
# =========================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    history.history[
        "accuracy"
    ],
    label="Train Accuracy"
)

plt.plot(
    history.history[
        "val_accuracy"
    ],
    label="Validation Accuracy"
)


plt.title(
    "V6 Training Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    ACCURACY_CURVE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 17. Loss Curve
# =========================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    history.history[
        "loss"
    ],
    label="Train Loss"
)

plt.plot(
    history.history[
        "val_loss"
    ],
    label="Validation Loss"
)


plt.title(
    "V6 Training Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    LOSS_CURVE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 18. 完成
# =========================================================

print()
print("=" * 75)
print("V6 訓練完成")
print("=" * 75)

print()
print(
    f"Best Model:\n{MODEL_PATH}"
)

print()
print(
    f"Test Report:\n{REPORT_PATH}"
)

print()
print(
    f"Training History:\n{HISTORY_PATH}"
)

print()
print(
    f"Confusion Matrix:\n{CONFUSION_MATRIX_PATH}"
)

print()
print(
    f"Accuracy Curve:\n{ACCURACY_CURVE_PATH}"
)

print()
print(
    f"Loss Curve:\n{LOSS_CURVE_PATH}"
)