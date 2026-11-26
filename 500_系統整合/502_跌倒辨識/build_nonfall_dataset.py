from ultralytics import YOLO
import cv2
import os
import numpy as np

# ==========================================
# Build Non-Fall Dataset
#
# 單張圖片
# -> YOLO11 Pose
# -> 13 Keypoints
# -> Normalize
# -> 同一姿勢複製 30 Frames
# -> Non_Fall = 0
# ==========================================


# ==========================================
# 1. 基本設定
# ==========================================

IMAGE_FOLDER = "Non_Fall"

OUTPUT_FILE = "nonfall_dataset.npz"

SEQUENCE_LENGTH = 30


# ==========================================
# 2. YOLO11 Pose
# ==========================================

model = YOLO("yolo11n-pose.pt")


# ==========================================
# 3. 保留的 13 個骨架點
# ==========================================

SELECTED_POINTS = [
    0,

    5, 6,

    7, 8,

    9, 10,

    11, 12,

    13, 14,

    15, 16
]


# ==========================================
# 4. 取得圖片
# ==========================================

image_files = [
    file
    for file in os.listdir(IMAGE_FOLDER)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]


print(
    "Non-Fall 圖片數量：",
    len(image_files)
)


# ==========================================
# 5. Dataset
# ==========================================

X = []

y = []


# ==========================================
# 6. 每張圖片處理
# ==========================================

for image_id, image_name in enumerate(image_files):

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )


    frame = cv2.imread(
        image_path
    )


    if frame is None:

        print(
            "無法讀取：",
            image_name
        )

        continue


    # ======================================
    # YOLO Pose
    # ======================================

    results = model(
        frame,
        verbose=False
    )


    # ======================================
    # 沒有偵測到人體
    # ======================================

    if (
        len(results) == 0
        or results[0].keypoints is None
        or results[0].keypoints.conf is None
    ):

        print(
            "沒有偵測到人體：",
            image_name
        )

        continue


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


    if len(xy) == 0:

        continue


    # ======================================
    # 7. 取得 13 個骨架點
    # ======================================

    pose_data = []


    for idx in SELECTED_POINTS:

        if conf[0][idx] < 0.5:

            pose_data.extend(
                [-1.0, -1.0]
            )

        else:

            x = float(
                xy[0][idx][0]
            )

            y_value = float(
                xy[0][idx][1]
            )

            pose_data.extend(
                [x, y_value]
            )


    pose = np.array(
        pose_data,
        dtype=np.float32
    ).reshape(
        13,
        2
    )


    # ======================================
    # 8. 檢查 Hip
    # ======================================

    left_hip = pose[7]

    right_hip = pose[8]


    if (
        -1 in left_hip
        or -1 in right_hip
    ):

        print(
            "Hip 無法偵測：",
            image_name
        )

        continue


    # ======================================
    # 9. Hip Center
    # ======================================

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


    # ======================================
    # 10. Shoulder Distance Normalize
    # ======================================

    left_shoulder = normalized_pose[1]

    right_shoulder = normalized_pose[2]


    shoulder_distance = np.linalg.norm(
        left_shoulder
        -
        right_shoulder
    )


    if shoulder_distance < 1e-6:

        print(
            "肩膀距離異常：",
            image_name
        )

        continue


    normalized_pose = (
        normalized_pose
        /
        shoulder_distance
    )


    # 13 × 2 -> 26
    normalized_pose = (
        normalized_pose
        .reshape(26)
    )


    # ======================================
    # 11. 同一姿勢複製 30 次
    # ======================================
    #
    # shape:
    #
    # (30, 26)
    #
    # ======================================

    sequence = np.tile(
        normalized_pose,
        (
            SEQUENCE_LENGTH,
            1
        )
    )


    # ======================================
    # 12. Non_Fall Label
    # ======================================

    X.append(
        sequence
    )

    y.append(
        0
    )


    print(
        f"{image_id + 1}/{len(image_files)}",
        image_name
    )


# ==========================================
# 13. NumPy
# ==========================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int64
)


# ==========================================
# 14. 統計
# ==========================================

print()
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
    "Non-Fall sequences：",
    len(y)
)

print("==============================")


# ==========================================
# 15. 儲存
# ==========================================

np.savez(
    OUTPUT_FILE,
    X=X,
    y=y
)


print(
    "Dataset 已儲存：",
    OUTPUT_FILE
)