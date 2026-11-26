from ultralytics import YOLO
import cv2
import os
import re
import numpy as np

# ==========================================
# Build Fall Dataset
#
# Continuous Images
# -> Read Real Frame Number
# -> YOLO11 Pose
# -> 13 Keypoints
# -> Missing Value Interpolation
# -> Normalize
# -> 30-frame Sequence
# -> Fall / Non_Fall
# ==========================================


# ==========================================
# 1. 基本設定
# ==========================================

IMAGE_FOLDER = "fall11"

OUTPUT_FILE = "fall11_dataset.npz"

# LSTM 每次看 30 張連續圖片
SEQUENCE_LENGTH = 30

# Sliding Window 每次往後移 5 張
STEP_SIZE = 5

# 一個 sequence 中
# 至少包含幾張真正跌倒 frame
# 才標記為 Fall
MIN_FALL_FRAMES = 5


# ==========================================
# 2. 真正跌倒的 Frame 區間
# ==========================================
#
# 注意：
# 這裡使用的是圖片檔名中的真正 Frame Number
#
# 例如：
# frame_00066.jpg ~ frame_00080.jpg
#
# 就寫：
# (66, 80)
#
# 如果同一段有多次跌倒：
#
# FALL_RANGES = [
#     (66, 80),
#     (100, 115),
# ]
#
# 請依實際圖片修改
# ==========================================

FALL_RANGES = [
    (230, 272),
]


# ==========================================
# 3. 從圖片檔名取得真正 Frame 編號
# ==========================================
#
# frame_00050.jpg -> 50
# frame_00107.jpg -> 107
# ==========================================

'''def get_frame_number(filename):

    match = re.search(
        r"frame_(\d+)",
        filename
    )

    if match:

        return int(
            match.group(1)
        )

    return None'''

# ==========================================
# 取得所有圖片
# ==========================================

image_files = [
    file
    for file in os.listdir(IMAGE_FOLDER)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

# 按照檔名排序
# timestamp 格式固定，因此排序後就是時間順序
image_files.sort()

print("==============================")
print("圖片數量：", len(image_files))
print("==============================")

# ==========================================
# 4. 載入 YOLO11 Pose
# ==========================================

model = YOLO(
    "yolo11n-pose.pt"
)

for frame_id, image_name in enumerate(image_files):

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    frame = cv2.imread(image_path)

    if frame is None:
        print("無法讀取：", image_name)
        continue

    print(
        f"Frame {frame_id:03d} | {image_name}"
    )


# ==========================================
# 5. 保留的 13 個骨架點
# ==========================================
#
# YOLO Pose COCO：
#
# 0  nose
#
# 5  left shoulder
# 6  right shoulder
#
# 7  left elbow
# 8  right elbow
#
# 9  left wrist
# 10 right wrist
#
# 11 left hip
# 12 right hip
#
# 13 left knee
# 14 right knee
#
# 15 left ankle
# 16 right ankle
#
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
# 6. 取得所有圖片
# ==========================================

image_files = [
    file
    for file in os.listdir(
        IMAGE_FOLDER
    )
    if file.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png"
        )
    )
]


# ==========================================
# 7. 只保留能取得 Frame Number 的圖片
# ==========================================
'''
image_files = [
    file
    for file in image_files
    if get_frame_number(file) is not None
]
'''


# ==========================================
# 8. 按真正 Frame Number 排序
# ==========================================
'''
image_files.sort(
    key=get_frame_number
)
'''

# timestamp 檔名按照時間排序
image_files.sort()

print()
print("==============================")
print("圖片數量：", len(image_files))
print("==============================")


# ==========================================
# 9. 儲存骨架和真正 Frame Number
# ==========================================

all_frames = []

frame_numbers = []


# ==========================================
# 10. 每張圖片執行 YOLO Pose
# ==========================================

for image_index, image_name in enumerate(
    image_files
):

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

'''
    # 真正的 Frame Number
    frame_number = get_frame_number(
        image_name
    )
'''

for frame_number, file in enumerate(image_files):

    image_path = os.path.join(
        IMAGE_FOLDER,
        file
    )

    # --------------------------------------
    # 讀取圖片
    # --------------------------------------

    frame = cv2.imread(
        image_path
    )


    # ======================================
    # 圖片讀取失敗
    # ======================================

    if frame is None:

        print(
            "無法讀取：",
            image_name
        )

        # 保留這一幀的位置
        all_frames.append(
            [-1.0] * 26
        )

        frame_numbers.append(
            frame_number
        )

        continue


    # ======================================
    # YOLO11 Pose
    # ======================================

    results = model(
        frame,
        verbose=False
    )


    # 預設：
    # 13 個 keypoints × x,y
    # 全部 missing
    pose_data = [
        -1.0
    ] * 26


    # ======================================
    # 11. 檢查是否有偵測到人體骨架
    # ======================================

    if (
        len(results) > 0
        and results[0].keypoints is not None
        and results[0].keypoints.conf is not None
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


            # 目前只取第一個人
            for idx in SELECTED_POINTS:

                # --------------------------
                # 信心值太低
                # --------------------------

                if conf[0][idx] < 0.5:

                    pose_data.extend(
                        [-1.0, -1.0]
                    )


                else:

                    x = float(
                        xy[0][idx][0]
                    )

                    y = float(
                        xy[0][idx][1]
                    )

                    pose_data.extend(
                        [x, y]
                    )


    # ======================================
    # 12. 儲存這一幀
    # ======================================

    all_frames.append(
        pose_data
    )

    frame_numbers.append(
        frame_number
    )


    print(
        f"{image_index + 1}/{len(image_files)}"
        f" | Frame {frame_number}"
        f" | {image_name}"
    )


# ==========================================
# 13. 轉成 NumPy
# ==========================================

all_frames = np.array(
    all_frames,
    dtype=np.float32
)

frame_numbers = np.array(
    frame_numbers,
    dtype=np.int32
)


print()
print("==============================")

print(
    "原始骨架 Shape：",
    all_frames.shape
)

print(
    "Frame Number 範圍：",
    frame_numbers[0],
    "~",
    frame_numbers[-1]
)

print("==============================")


# ==========================================
# 14. Missing Keypoints 插值
# ==========================================
#
# -1 代表：
# YOLO 沒有可靠偵測到這個點
#
# 使用前後 frame 線性插值
# ==========================================

for feature_index in range(
    all_frames.shape[1]
):

    values = all_frames[
        :,
        feature_index
    ]


    valid_mask = (
        values != -1
    )


    valid_indices = np.where(
        valid_mask
    )[0]


    # 至少需要 2 個有效資料
    if len(valid_indices) >= 2:

        missing_indices = np.where(
            ~valid_mask
        )[0]


        values[
            missing_indices
        ] = np.interp(

            missing_indices,

            valid_indices,

            values[
                valid_indices
            ]
        )


        all_frames[
            :,
            feature_index
        ] = values


# ==========================================
# 15. Normalization
# ==========================================
#
# 1. Hip Center 當中心
#
# 2. Shoulder Distance 當 Scale
#
# ==========================================

normalized_frames = []


for pose in all_frames:

    pose = pose.reshape(
        13,
        2
    )


    # ======================================
    # 左右 Hip
    # ======================================

    left_hip = pose[7]

    right_hip = pose[8]


    # ======================================
    # Hip 還是 missing
    # ======================================

    if (
        -1 in left_hip
        or
        -1 in right_hip
    ):

        normalized_frames.append(
            np.zeros(
                (13, 2),
                dtype=np.float32
            )
        )

        continue


    # ======================================
    # 16. Hip Center
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
    # 17. Shoulder Distance
    # ======================================

    left_shoulder = normalized_pose[1]

    right_shoulder = normalized_pose[2]


    shoulder_distance = np.linalg.norm(
        left_shoulder
        -
        right_shoulder
    )


    # 避免除以 0
    if shoulder_distance < 1e-6:

        shoulder_distance = 1.0


    # ======================================
    # Scale Normalization
    # ======================================

    normalized_pose = (
        normalized_pose
        /
        shoulder_distance
    )


    normalized_frames.append(
        normalized_pose
    )


# ==========================================
# 18. 轉成 NumPy
# ==========================================

normalized_frames = np.array(
    normalized_frames,
    dtype=np.float32
)


# 13 × 2 -> 26
normalized_frames = normalized_frames.reshape(
    len(normalized_frames),
    26
)


print(
    "Normalization Shape：",
    normalized_frames.shape
)


# ==========================================
# 19. 建立 Sliding Window
# ==========================================

X = []

y = []

sequence_ranges = []

fall_frame_counts = []


for start_index in range(
    0,
    len(normalized_frames)
    - SEQUENCE_LENGTH
    + 1,
    STEP_SIZE
):

    # ======================================
    # Sequence Index
    # ======================================

    end_index = (
        start_index
        +
        SEQUENCE_LENGTH
        -
        1
    )


    # ======================================
    # 取出 30 個 Pose Frames
    # ======================================

    sequence = normalized_frames[
        start_index:
        start_index + SEQUENCE_LENGTH
    ]


    # ======================================
    # 對應真正 Frame Number
    # ======================================

    sequence_frame_numbers = frame_numbers[
        start_index:
        start_index + SEQUENCE_LENGTH
    ]


    start_frame = int(
        sequence_frame_numbers[0]
    )

    end_frame = int(
        sequence_frame_numbers[-1]
    )


    # ======================================
    # 20. 計算 Sequence 中
    #     有多少真正 Fall Frame
    # ======================================

    fall_frame_count = 0


    for current_frame in sequence_frame_numbers:

        is_fall_frame = False


        for fall_start, fall_end in FALL_RANGES:

            if (
                fall_start
                <= current_frame
                <= fall_end
            ):

                is_fall_frame = True

                break


        if is_fall_frame:

            fall_frame_count += 1


    # ======================================
    # 21. Fall / Non_Fall
    # ======================================

    if fall_frame_count >= MIN_FALL_FRAMES:

        label = 1

    else:

        label = 0


    # ======================================
    # 22. 儲存 Sequence
    # ======================================

    X.append(
        sequence
    )

    y.append(
        label
    )

    sequence_ranges.append(
        [
            start_frame,
            end_frame
        ]
    )

    fall_frame_counts.append(
        fall_frame_count
    )


# ==========================================
# 23. 轉成 NumPy
# ==========================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int64
)

sequence_ranges = np.array(
    sequence_ranges,
    dtype=np.int32
)

fall_frame_counts = np.array(
    fall_frame_counts,
    dtype=np.int32
)


# ==========================================
# 24. Dataset 統計
# ==========================================

fall_count = np.sum(
    y == 1
)

non_fall_count = np.sum(
    y == 0
)


print()
print("==============================")
print("Dataset 統計")
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
    "Fall sequences：",
    fall_count
)

print(
    "Non-Fall sequences：",
    non_fall_count
)

print("==============================")


# ==========================================
# 25. 顯示每一組 Sequence
# ==========================================

print()
print("Sequence Label：")
print()


for i in range(
    len(y)
):

    start_frame = (
        sequence_ranges[i][0]
    )

    end_frame = (
        sequence_ranges[i][1]
    )

    fall_frames = (
        fall_frame_counts[i]
    )


    if y[i] == 1:

        label_name = "Fall"

    else:

        label_name = "Non_Fall"


    print(

        f"{start_frame:05d}"
        f" ~ "
        f"{end_frame:05d}"

        f" | Fall Frames = "
        f"{fall_frames:02d}"

        f" | {label_name}"
    )


# ==========================================
# 26. 儲存 Dataset
# ==========================================

np.savez(

    OUTPUT_FILE,

    X=X,

    y=y,

    ranges=sequence_ranges,

    fall_frame_counts=fall_frame_counts,

    frame_numbers=frame_numbers
)


print()
print("==============================")

print(
    "Dataset 已儲存：",
    OUTPUT_FILE
)

print("==============================")