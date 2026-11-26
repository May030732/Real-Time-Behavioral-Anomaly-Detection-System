from ultralytics import YOLO
import cv2
import numpy as np


# ==========================================
# Video -> Non-Fall LSTM Dataset
#
# 功能：
# 背面.mp4
# -> YOLO11 Pose
# -> 13 Keypoints
# -> Normalize
# -> 30-frame Sequence
# -> Non_Fall = 0
# ==========================================


# ==========================================
# 1. 基本設定
# ==========================================

VIDEO_FILE = "躺著.mp4"

OUTPUT_FILE = "nonfall_lying_dataset.npz"

# LSTM 一次看 30 frames
SEQUENCE_LENGTH = 30

# Sliding Window 每次往後移 5 frames
STEP_SIZE = 5

# YOLO 骨架點信心值門檻
KEYPOINT_CONF_THRESHOLD = 0.5


# ==========================================
# 2. 載入 YOLO11 Pose
# ==========================================

model = YOLO(
    "yolo11n-pose.pt"
)


# ==========================================
# 3. 保留 13 個骨架點
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
# 4. Pose Normalization
# ==========================================
#
# 注意：
# 這裡使用跟你目前模型相同的方式：
#
# 1. Hip Center 當中心
# 2. Shoulder Distance 當 scale
#
# 因為目前只是「補側身資料」
# 還沒有重新改整套 normalization
#
# ==========================================

def normalize_pose(pose):

    # --------------------------------------
    # 左右 Hip
    # --------------------------------------

    left_hip = pose[7]
    right_hip = pose[8]


    # Hip 無法偵測
    if (
        -1 in left_hip
        or
        -1 in right_hip
    ):

        return None


    # ======================================
    # Hip Center
    # ======================================

    center = (
        left_hip
        +
        right_hip
    ) / 2.0


    # 所有骨架點移到 Hip Center
    normalized_pose = (
        pose
        -
        center
    )


    # ======================================
    # Shoulder Distance
    # ======================================

    left_shoulder = normalized_pose[1]
    right_shoulder = normalized_pose[2]


    shoulder_distance = np.linalg.norm(
        left_shoulder
        -
        right_shoulder
    )


    # 防止除以 0
    if shoulder_distance < 1e-6:

        return None


    # ======================================
    # Scale Normalization
    # ======================================

    normalized_pose = (
        normalized_pose
        /
        shoulder_distance
    )


    # 13 × 2 -> 26
    return (
        normalized_pose
        .reshape(26)
        .astype(np.float32)
    )


# ==========================================
# 5. 開啟影片
# ==========================================

cap = cv2.VideoCapture(
    VIDEO_FILE
)


if not cap.isOpened():

    print(
        "影片開啟失敗：",
        VIDEO_FILE
    )

    exit()


# ==========================================
# 6. 儲存所有有效骨架 Frame
# ==========================================

all_frames = []

# 上一個有效 Pose
# 用來補某些暫時抓不到的骨架點
previous_pose = None

frame_id = 0


# ==========================================
# 7. 一幀一幀處理影片
# ==========================================

while True:

    ret, frame = cap.read()


    if not ret:

        break


    # ======================================
    # YOLO Pose
    # ======================================

    results = model(
        frame,
        verbose=False
    )


    # ======================================
    # 是否有偵測到人體
    # ======================================

    if (
        len(results) > 0
        and
        results[0].keypoints is not None
        and
        results[0].keypoints.conf is not None
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
        # 至少偵測到一個人
        # ==================================

        if len(xy) > 0:

            pose_data = []


            # ==============================
            # 取得 13 個骨架點
            # ==============================

            for point_index, idx in enumerate(
                SELECTED_POINTS
            ):

                # --------------------------
                # 信心值低於門檻
                # --------------------------

                if (
                    conf[0][idx]
                    <
                    KEYPOINT_CONF_THRESHOLD
                ):

                    # 有上一幀
                    # 就使用上一幀的同一個點
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
            # 儲存上一幀 Pose
            # ==================================

            previous_pose = (
                pose_data.copy()
            )


            # ==================================
            # Normalize
            # ==================================

            normalized_pose = normalize_pose(
                pose_data
            )


            # ==================================
            # 有效資料才加入
            # ==================================

            if normalized_pose is not None:

                all_frames.append(
                    normalized_pose
                )


    frame_id += 1


    print(
        f"Processing frame: {frame_id}"
    )


# ==========================================
# 8. 關閉影片
# ==========================================

cap.release()


# ==========================================
# 9. 轉成 NumPy
# ==========================================

all_frames = np.array(
    all_frames,
    dtype=np.float32
)


print()
print("==============================")

print(
    "有效 Pose Frames：",
    all_frames.shape
)

print("==============================")


# ==========================================
# 10. 檢查是否至少有 30 frames
# ==========================================

if len(all_frames) < SEQUENCE_LENGTH:

    print(
        "有效 Frame 不足",
        SEQUENCE_LENGTH,
        "張，無法建立 LSTM Sequence"
    )

    exit()


# ==========================================
# 11. Sliding Window
# ==========================================

X = []

y = []


for start in range(
    0,
    len(all_frames)
    - SEQUENCE_LENGTH
    + 1,
    STEP_SIZE
):

    # --------------------------------------
    # 取 30 frames
    # --------------------------------------

    sequence = all_frames[
        start:
        start + SEQUENCE_LENGTH
    ]


    # --------------------------------------
    # 儲存 Sequence
    # --------------------------------------

    X.append(
        sequence
    )


    # --------------------------------------
    # 0 = Non_Fall
    # --------------------------------------

    y.append(
        0
    )


# ==========================================
# 12. 轉成 NumPy
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
# 13. 顯示 Dataset 統計
# ==========================================

print()
print("==============================")
print("Non-Fall Turn Dataset")
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
# 14. 儲存 Dataset
# ==========================================

np.savez(

    OUTPUT_FILE,

    X=X,

    y=y
)


print()
print(
    "Dataset 已儲存：",
    OUTPUT_FILE
)