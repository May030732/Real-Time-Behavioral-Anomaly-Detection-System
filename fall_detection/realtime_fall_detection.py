from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf
from collections import deque


# ==========================================
# 1. 基本設定
# ==========================================

POSE_MODEL_FILE = "yolo11n-pose.pt"
LSTM_MODEL_FILE = "fall_lstm_model.keras"

SEQUENCE_LENGTH = 30

# LSTM 輸出 >= 0.5 判定 Fall
FALL_THRESHOLD = 0.7

# YOLO Keypoint 信心值門檻
KEYPOINT_CONF_THRESHOLD = 0.5


# ==========================================
# 2. 載入模型
# ==========================================

print("Loading YOLO Pose model...")

pose_model = YOLO(
    POSE_MODEL_FILE
)

print("Loading LSTM model...")

lstm_model = tf.keras.models.load_model(
    LSTM_MODEL_FILE
)

print("Models loaded successfully.")


# ==========================================
# 3. 保留的 13 個骨架點
# ==========================================

SELECTED_POINTS = [
    0,       # nose

    5, 6,    # shoulders

    7, 8,    # elbows

    9, 10,   # wrists

    11, 12,  # hips

    13, 14,  # knees

    15, 16   # ankles
]


# ==========================================
# 4. 建立 30 Frame Buffer
# ==========================================

sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)


# ==========================================
# 5. 儲存上一幀有效骨架
#
# 如果這一幀部分點抓不到，
# 第一版先用上一幀補
# ==========================================

previous_pose = None


# ==========================================
# 6. 開啟攝影機
# ==========================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

print(
    "Camera Open:",
    cap.isOpened()
)


if not cap.isOpened():

    print(
        "攝影機開啟失敗"
    )

    exit()


# ==========================================
# 7. Pose 正規化函式
# ==========================================

def normalize_pose(pose):

    """
    pose shape:
    (13, 2)

    做法與訓練資料一致：
    1. Hip Center
    2. Shoulder Distance
    """

    # --------------------------------------
    # 左右 Hip
    # --------------------------------------

    left_hip = pose[7]
    right_hip = pose[8]


    # Hip 無效
    if (
        -1 in left_hip
        or
        -1 in right_hip
    ):

        return None


    # --------------------------------------
    # Hip Center
    # --------------------------------------

    center = (
        left_hip
        +
        right_hip
    ) / 2.0


    normalized_pose = (
        pose
        -
        center
    )


    # --------------------------------------
    # Shoulder Distance
    # --------------------------------------

    left_shoulder = (
        normalized_pose[1]
    )

    right_shoulder = (
        normalized_pose[2]
    )


    shoulder_distance = np.linalg.norm(
        left_shoulder
        -
        right_shoulder
    )


    # 避免異常
    if shoulder_distance < 1e-6:

        return None


    # --------------------------------------
    # Scale Normalization
    # --------------------------------------

    normalized_pose = (
        normalized_pose
        /
        shoulder_distance
    )


    return (
        normalized_pose
        .reshape(26)
        .astype(np.float32)
    )


# ==========================================
# 8. 即時辨識
# ==========================================

while True:

    ret, frame = cap.read()


    if not ret:

        print(
            "讀不到攝影機畫面"
        )

        break


    # ======================================
    # YOLO Pose
    # ======================================

    results = pose_model(
        frame,
        verbose=False
    )


    annotated = results[0].plot()


    # 預設狀態
    status = "Collecting..."

    probability = 0.0


    # ======================================
    # 有偵測到人體
    # ======================================

    if (
        len(results) > 0
        and
        results[0].keypoints
        is not None
        and
        results[0].keypoints.conf
        is not None
    ):

        xy = (
            results[0]
            .keypoints
            .xy
            .cpu()
            .numpy()
        )


        conf = (
            results[0]
            .keypoints
            .conf
            .cpu()
            .numpy()
        )


        # ==================================
        # 至少一個人
        # ==================================

        if len(xy) > 0:

            pose_data = []


            # ------------------------------
            # 取得 13 個骨架點
            # ------------------------------

            for point_index, idx in enumerate(
                SELECTED_POINTS
            ):

                if (
                    conf[0][idx]
                    <
                    KEYPOINT_CONF_THRESHOLD
                ):

                    # ----------------------
                    # 如果這一點抓不到
                    # 有上一幀就用上一幀
                    # ----------------------

                    if previous_pose is not None:

                        pose_data.append(
                            previous_pose[
                                point_index
                            ].copy()
                        )

                    else:

                        pose_data.append(
                            np.array(
                                [-1.0, -1.0],
                                dtype=np.float32
                            )
                        )


                else:

                    x = float(
                        xy[0][idx][0]
                    )

                    y = float(
                        xy[0][idx][1]
                    )


                    pose_data.append(
                        np.array(
                            [x, y],
                            dtype=np.float32
                        )
                    )


            pose_data = np.array(
                pose_data,
                dtype=np.float32
            )


            # ==================================
            # 儲存成上一幀
            # ==================================

            previous_pose = (
                pose_data.copy()
            )


            # ==================================
            # Normalization
            # ==================================

            normalized_pose = normalize_pose(
                pose_data
            )


            if normalized_pose is not None:

                sequence_buffer.append(
                    normalized_pose
                )


    # ======================================
    # Buffer 滿 30 Frames
    # 才開始 LSTM 預測
    # ======================================

    if (
        len(sequence_buffer)
        ==
        SEQUENCE_LENGTH
    ):

        sequence = np.array(
            sequence_buffer,
            dtype=np.float32
        )


        # 加 batch dimension
        #
        # (30,26)
        # ->
        # (1,30,26)

        sequence = np.expand_dims(
            sequence,
            axis=0
        )


        # ==================================
        # LSTM 預測
        # ==================================

        prediction = lstm_model.predict(
            sequence,
            verbose=0
        )


        probability = float(
            prediction[0][0]
        )


        # ==================================
        # Fall / Non-Fall
        # ==================================

        if (
            probability
            >=
            FALL_THRESHOLD
        ):

            status = "FALL"

        else:

            status = "NON_FALL"


    # ======================================
    # 9. 顯示 Buffer 狀態
    # ======================================

    cv2.putText(
        annotated,

        f"Frames: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",

        (20, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 255),

        2
    )


    # ======================================
    # 10. 顯示結果
    # ======================================

    if status == "FALL":

        text_color = (
            0,
            0,
            255
        )

    else:

        text_color = (
            0,
            255,
            0
        )


    cv2.putText(
        annotated,

        f"Status: {status}",

        (20, 65),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        text_color,

        2
    )


    # ======================================
    # 11. 顯示 Fall Probability
    # ======================================

    if (
        len(sequence_buffer)
        ==
        SEQUENCE_LENGTH
    ):

        cv2.putText(
            annotated,

            f"Fall probability: {probability:.3f}",

            (20, 100),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            text_color,

            2
        )


    # ======================================
    # 12. 顯示畫面
    # ======================================

    cv2.imshow(
        "Real-Time Fall Detection",
        annotated
    )


    # ======================================
    # 13. 按 q 離開
    # ======================================

    if (
        cv2.waitKey(1)
        &
        0xFF
        ==
        ord("q")
    ):

        break


# ==========================================
# 14. 關閉
# ==========================================

cap.release()

cv2.destroyAllWindows()

print(
    "Real-Time Fall Detection Closed."
)