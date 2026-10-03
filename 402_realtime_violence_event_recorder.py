from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf

from collections import deque
from datetime import datetime
from pathlib import Path
import time


# =========================================================
# 1. 基本設定
# =========================================================

POSE_MODEL_FILE = "yolo11n-pose.pt"

V6_MODEL_FILE = Path(
    r"C:\Users\USER\Desktop\學校功課\專題\101_vedio_pose_to_csv"
    r"\10119_LSTM模型(第六版_真人偏差修正)"
    r"\lstm_v6_dual_branch.keras"
)

VIDEO_SAVE_FOLDER = Path(
    r"C:\Users\USER\Desktop\學校功課\專題"
    r"\violence_videos"
)

VIDEO_SAVE_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


CAMERA_INDEX = 0

SEQUENCE_LENGTH = 30

KEYPOINT_CONF_THRESHOLD = 0.5

VIOLENCE_THRESHOLD = 0.5


# =========================================================
# 2. 異常事件影片設定
# =========================================================

# 異常前 5 秒
PRE_EVENT_SECONDS = 5.0

# 異常後 5 秒
POST_EVENT_SECONDS = 5.0

# 一支影片完成後至少等待 10 秒
# 才允許下一次事件開始
RECORD_COOLDOWN_SECONDS = 10.0


# =========================================================
# 3. 13 個 Keypoints
# =========================================================

SELECTED_POINTS = [
    0,      # nose

    5,      # left shoulder
    6,      # right shoulder

    7,      # left elbow
    8,      # right elbow

    9,      # left wrist
    10,     # right wrist

    11,     # left hip
    12,     # right hip

    13,     # left knee
    14,     # right knee

    15,     # left ankle
    16      # right ankle
]


# =========================================================
# 4. 13 點內部索引
# =========================================================

LEFT_SHOULDER = 1
RIGHT_SHOULDER = 2

LEFT_HIP = 7
RIGHT_HIP = 8


# =========================================================
# 5. V6 Normalization 設定
# =========================================================

MIN_BODY_SCALE = 5.0

CLIP_MIN = -20.0
CLIP_MAX = 20.0


# =========================================================
# 6. 載入模型
# =========================================================

print("=" * 70)
print("Loading Models")
print("=" * 70)


print("Loading YOLO11 Pose...")

pose_model = YOLO(
    POSE_MODEL_FILE
)


print("Loading V6 Dual-Branch LSTM...")

if not V6_MODEL_FILE.exists():

    raise FileNotFoundError(
        f"找不到 V6 模型：{V6_MODEL_FILE}"
    )


violence_model = tf.keras.models.load_model(
    V6_MODEL_FILE
)


print("Models loaded successfully.")

print()
print("V6 Input Shapes:")

for input_tensor in violence_model.inputs:

    print(
        input_tensor.name,
        input_tensor.shape
    )


# =========================================================
# 7. LSTM Buffer
# =========================================================

pose_sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)

motion_sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)


previous_normalized_pose = None
previous_valid_mask = None


# =========================================================
# 8. 影片 Buffer / Event 狀態
# =========================================================

# 平常只在 RAM 保存最近 5 秒畫面
#
# 格式：
# (time.monotonic(), frame)
#
video_buffer = deque()


# 是否正在收集「異常後 5 秒」
recording_event = False


# 最後真正要輸出的畫面
event_video_frames = []

event_video_times = []


# 後 5 秒截止時間
recording_end_time = 0.0


# 上一支影片完成時間
last_video_saved_time = (
    -RECORD_COOLDOWN_SECONDS
)


# 上一個辨識狀態
previous_status = "NON_VIOLENCE"


# =========================================================
# 9. 人體中心
# =========================================================

def get_body_center(
    points,
    valid
):

    if (
        valid[LEFT_HIP]
        and
        valid[RIGHT_HIP]
    ):

        return (
            points[LEFT_HIP]
            + points[RIGHT_HIP]
        ) / 2.0


    if (
        valid[LEFT_SHOULDER]
        and
        valid[RIGHT_SHOULDER]
    ):

        return (
            points[LEFT_SHOULDER]
            + points[RIGHT_SHOULDER]
        ) / 2.0


    return np.mean(
        points[valid],
        axis=0
    )


# =========================================================
# 10. 人體 Scale
# =========================================================

def get_body_scale(
    points,
    valid,
    center
):

    # -----------------------------------------------------
    # Shoulder width
    # -----------------------------------------------------

    if (
        valid[LEFT_SHOULDER]
        and
        valid[RIGHT_SHOULDER]
    ):

        shoulder_width = np.linalg.norm(
            points[LEFT_SHOULDER]
            - points[RIGHT_SHOULDER]
        )


        if shoulder_width >= MIN_BODY_SCALE:

            return float(
                shoulder_width
            )


    # -----------------------------------------------------
    # Hip width
    # -----------------------------------------------------

    if (
        valid[LEFT_HIP]
        and
        valid[RIGHT_HIP]
    ):

        hip_width = np.linalg.norm(
            points[LEFT_HIP]
            - points[RIGHT_HIP]
        )


        if hip_width >= MIN_BODY_SCALE:

            return float(
                hip_width
            )


    # -----------------------------------------------------
    # Bounding Box
    # -----------------------------------------------------

    valid_points = points[
        valid
    ]


    if len(valid_points) >= 2:

        min_xy = np.min(
            valid_points,
            axis=0
        )

        max_xy = np.max(
            valid_points,
            axis=0
        )


        width = (
            max_xy[0]
            - min_xy[0]
        )

        height = (
            max_xy[1]
            - min_xy[1]
        )


        bbox_scale = max(
            float(width),
            float(height)
        )


        if bbox_scale >= MIN_BODY_SCALE:

            return bbox_scale


    # -----------------------------------------------------
    # Maximum radius
    # -----------------------------------------------------

    distances = np.linalg.norm(
        valid_points - center,
        axis=1
    )


    if len(distances) > 0:

        max_distance = float(
            np.max(distances)
        )


        if max_distance >= MIN_BODY_SCALE:

            return max_distance


    return MIN_BODY_SCALE


# =========================================================
# 11. Pose Normalization
# =========================================================

def normalize_pose(
    raw_pose
):

    points = np.array(
        raw_pose,
        dtype=np.float32
    ).reshape(
        13,
        2
    )


    valid = ~np.all(
        points == 0,
        axis=1
    )


    if not np.any(valid):

        return (
            np.zeros(
                26,
                dtype=np.float32
            ),
            valid
        )


    center = get_body_center(
        points,
        valid
    )


    scale = get_body_scale(
        points,
        valid,
        center
    )


    normalized = np.zeros_like(
        points,
        dtype=np.float32
    )


    normalized[valid] = (
        points[valid]
        - center
    ) / scale


    normalized[valid] = np.clip(
        normalized[valid],
        CLIP_MIN,
        CLIP_MAX
    )


    normalized[~valid] = 0.0


    return (
        normalized.reshape(26),
        valid
    )


# =========================================================
# 12. Motion
# =========================================================

def calculate_motion(
    current_pose,
    current_valid,
    previous_pose,
    previous_valid
):

    # 第一幀 Motion = 0
    if (
        previous_pose is None
        or
        previous_valid is None
    ):

        return np.zeros(
            26,
            dtype=np.float32
        )


    current_points = current_pose.reshape(
        13,
        2
    )

    previous_points = previous_pose.reshape(
        13,
        2
    )


    motion = np.zeros(
        (
            13,
            2
        ),
        dtype=np.float32
    )


    # 前後兩幀都有該 joint 才算 Motion
    valid_both = (
        current_valid
        & previous_valid
    )


    motion[
        valid_both
    ] = (
        current_points[
            valid_both
        ]
        - previous_points[
            valid_both
        ]
    )


    return motion.reshape(
        26
    )


# =========================================================
# 13. 找主要人物
# =========================================================

def get_person_area(
    person
):

    valid_points = []


    for kp_index in SELECTED_POINTS:

        x, y, conf = person[
            kp_index
        ]


        if conf >= KEYPOINT_CONF_THRESHOLD:

            valid_points.append(
                [
                    x,
                    y
                ]
            )


    if len(valid_points) < 2:

        return 0.0


    valid_points = np.array(
        valid_points,
        dtype=np.float32
    )


    min_xy = np.min(
        valid_points,
        axis=0
    )

    max_xy = np.max(
        valid_points,
        axis=0
    )


    return float(
        (
            max_xy[0]
            - min_xy[0]
        )
        *
        (
            max_xy[1]
            - min_xy[1]
        )
    )


# =========================================================
# 14. YOLO → Raw 13 Point Pose
# =========================================================

def extract_raw_pose(
    person
):

    raw_pose = []


    for kp_index in SELECTED_POINTS:

        x, y, conf = person[
            kp_index
        ]


        if conf >= KEYPOINT_CONF_THRESHOLD:

            raw_pose.extend(
                [
                    float(x),
                    float(y)
                ]
            )


        else:

            # V6 規則：
            # missing joint = (0, 0)

            raw_pose.extend(
                [
                    0.0,
                    0.0
                ]
            )


    return np.array(
        raw_pose,
        dtype=np.float32
    )


# =========================================================
# 15. 儲存 10 秒異常影片
# =========================================================

def save_event_video(
    frames,
    frame_times
):

    if len(frames) < 2:

        print(
            "影片畫面不足，無法儲存。"
        )

        return None


    # -----------------------------------------------------
    # 根據實際取得 Frame 的速度計算 FPS
    # -----------------------------------------------------

    video_duration = (
        frame_times[-1]
        - frame_times[0]
    )


    actual_fps = (
        (len(frames) - 1)
        /
        max(
            video_duration,
            0.001
        )
    )


    actual_fps = max(
        1.0,
        min(
            actual_fps,
            60.0
        )
    )


    # -----------------------------------------------------
    # Filename
    # -----------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    video_path = (
        VIDEO_SAVE_FOLDER
        / f"violence_{timestamp}.mp4"
    )


    # -----------------------------------------------------
    # Frame size
    # -----------------------------------------------------

    frame_height, frame_width = (
        frames[0].shape[:2]
    )


    # -----------------------------------------------------
    # Video Writer
    # -----------------------------------------------------

    writer = cv2.VideoWriter(
        str(video_path),

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        actual_fps,

        (
            frame_width,
            frame_height
        )
    )


    if not writer.isOpened():

        print(
            "影片建立失敗。"
        )

        return None


    # -----------------------------------------------------
    # Write
    # -----------------------------------------------------

    for saved_frame in frames:

        writer.write(
            saved_frame
        )


    writer.release()


    print()
    print("=" * 70)
    print("異常事件影片儲存完成")
    print("=" * 70)

    print(
        f"影片：{video_path}"
    )

    print(
        f"實際長度：約 {video_duration:.2f} 秒"
    )

    print(
        f"FPS：約 {actual_fps:.2f}"
    )


    return str(
        video_path
    )


# =========================================================
# 16. 上傳函式
# =========================================================

def upload_event_video(
    video_path
):

    """
    目前先保留上傳介面。

    video_path 就是剛剛產生的 10 秒 MP4 完整路徑。

    例如之後若使用 Discord：
        upload_event_video(video_path)

    就在這個函式內加入 Discord Webhook/API。
    """


    if video_path is None:

        return


    print()
    print(
        "影片已準備好，可進行上傳："
    )

    print(
        video_path
    )


    # =====================================================
    # TODO：
    # 在這裡加入真正的上傳程式
    # =====================================================


# =========================================================
# 17. Webcam
# =========================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    raise RuntimeError(
        "無法開啟 Webcam"
    )


print()
print("=" * 70)
print("V6 Violence Event Recorder")
print("=" * 70)

print(
    "平時只在 RAM 保留最近 5 秒畫面"
)

print(
    "偵測 Violence 後，再收後 5 秒"
)

print(
    "完成後輸出約 10 秒 MP4"
)

print(
    "按 Q 離開"
)


# =========================================================
# 18. Main Loop
# =========================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print(
            "讀不到攝影機畫面"
        )

        break


    current_time = time.monotonic()


    # =====================================================
    # YOLO Pose
    # =====================================================

    results = pose_model(
        frame,
        verbose=False
    )


    result = results[0]


    # 顯示 YOLO Skeleton
    annotated = result.plot()


    selected_person = None


    if (
        result.keypoints is not None
        and
        result.keypoints.data is not None
        and
        len(result.keypoints.data) > 0
    ):

        people = (
            result.keypoints.data
            .cpu()
            .numpy()
        )


        largest_area = 0.0


        for person in people:

            area = get_person_area(
                person
            )


            if area > largest_area:

                largest_area = area

                selected_person = person


    # =====================================================
    # 預設辨識狀態
    # =====================================================

    status = "COLLECTING"

    violence_probability = 0.0


    # =====================================================
    # V6 Pose + Motion
    # =====================================================

    if selected_person is not None:

        raw_pose = extract_raw_pose(
            selected_person
        )


        normalized_pose, current_valid_mask = normalize_pose(
            raw_pose
        )


        motion = calculate_motion(
            current_pose=normalized_pose,
            current_valid=current_valid_mask,
            previous_pose=previous_normalized_pose,
            previous_valid=previous_valid_mask
        )


        pose_sequence_buffer.append(
            normalized_pose
        )


        motion_sequence_buffer.append(
            motion
        )


        previous_normalized_pose = (
            normalized_pose.copy()
        )

        previous_valid_mask = (
            current_valid_mask.copy()
        )


    # =====================================================
    # 30 Frames → V6 Prediction
    # =====================================================

    if (
        len(pose_sequence_buffer)
        == SEQUENCE_LENGTH
        and
        len(motion_sequence_buffer)
        == SEQUENCE_LENGTH
    ):

        pose_sequence = np.array(
            pose_sequence_buffer,
            dtype=np.float32
        )


        motion_sequence = np.array(
            motion_sequence_buffer,
            dtype=np.float32
        )


        pose_input = np.expand_dims(
            pose_sequence,
            axis=0
        )


        motion_input = np.expand_dims(
            motion_sequence,
            axis=0
        )


        prediction = violence_model.predict(
            [
                pose_input,
                motion_input
            ],
            verbose=0
        )


        violence_probability = float(
            prediction[0][0]
        )


        if (
            violence_probability
            >= VIOLENCE_THRESHOLD
        ):

            status = "VIOLENCE"


        else:

            status = "NON_VIOLENCE"


    # =====================================================
    # 顯示結果
    # =====================================================

    if status == "VIOLENCE":

        text_color = (
            0,
            0,
            255
        )


    elif status == "NON_VIOLENCE":

        text_color = (
            0,
            255,
            0
        )


    else:

        text_color = (
            0,
            255,
            255
        )


    cv2.putText(
        annotated,

        f"Status: {status}",

        (
            20,
            30
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        text_color,

        2
    )


    cv2.putText(
        annotated,

        (
            f"Violence Probability: "
            f"{violence_probability:.3f}"
        ),

        (
            20,
            65
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        text_color,

        2
    )


    cv2.putText(
        annotated,

        (
            f"Frames: "
            f"{len(pose_sequence_buffer)}"
            f"/{SEQUENCE_LENGTH}"
        ),

        (
            20,
            100
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (
            255,
            255,
            255
        ),

        2
    )


    # =====================================================
    # 19. 前 5 秒 RAM Buffer
    # =====================================================

    # 儲存的是有 Skeleton / Status 顯示的畫面
    current_video_frame = (
        annotated.copy()
    )


    if not recording_event:

        video_buffer.append(
            (
                current_time,
                current_video_frame
            )
        )


        # -------------------------------------------------
        # 永遠只留下最近 5 秒
        # -------------------------------------------------

        while (
            video_buffer
            and
            (
                current_time
                - video_buffer[0][0]
            )
            > PRE_EVENT_SECONDS
        ):

            video_buffer.popleft()


    # =====================================================
    # 20. 判斷是否為新的 Violence Event
    # =====================================================

    new_violence_event = (
        status == "VIOLENCE"
        and
        previous_status != "VIOLENCE"
    )


    cooldown_finished = (
        (
            current_time
            - last_video_saved_time
        )
        >= RECORD_COOLDOWN_SECONDS
    )


    # =====================================================
    # 21. Violence 觸發
    # =====================================================

    if (
        new_violence_event
        and
        not recording_event
        and
        cooldown_finished
    ):

        recording_event = True


        # -------------------------------------------------
        # 複製異常前 5 秒
        # -------------------------------------------------

        event_video_times = [
            saved_time
            for saved_time, saved_frame
            in video_buffer
        ]


        event_video_frames = [
            saved_frame.copy()
            for saved_time, saved_frame
            in video_buffer
        ]


        # -------------------------------------------------
        # 再收異常後 5 秒
        # -------------------------------------------------

        recording_end_time = (
            current_time
            + POST_EVENT_SECONDS
        )


        print()
        print("=" * 70)

        print(
            "偵測到 VIOLENCE！"
        )

        print(
            "正在保存異常前 5 秒 + 後 5 秒..."
        )

        print("=" * 70)


    # =====================================================
    # 22. 收集異常後 5 秒
    # =====================================================

    if recording_event:

        # 避免 Trigger 當下的 frame 重複
        if (
            not event_video_times
            or
            current_time
            >
            event_video_times[-1]
        ):

            event_video_times.append(
                current_time
            )

            event_video_frames.append(
                current_video_frame.copy()
            )


        # -------------------------------------------------
        # 後 5 秒完成
        # -------------------------------------------------

        if (
            current_time
            >= recording_end_time
        ):

            video_path = save_event_video(
                event_video_frames,
                event_video_times
            )


            # ---------------------------------------------
            # 上傳
            # ---------------------------------------------

            if video_path is not None:

                upload_event_video(
                    video_path
                )


            # ---------------------------------------------
            # Event 完成
            # ---------------------------------------------

            last_video_saved_time = (
                current_time
            )


            recording_event = False


            event_video_frames.clear()

            event_video_times.clear()

            video_buffer.clear()


    # =====================================================
    # 23. 更新上一個辨識狀態
    # =====================================================

    previous_status = status


    # =====================================================
    # 24. 顯示 Recording 狀態
    # =====================================================

    if recording_event:

        remaining = max(
            0.0,
            recording_end_time
            - current_time
        )


        cv2.putText(
            annotated,

            (
                f"EVENT RECORDING: "
                f"{remaining:.1f}s"
            ),

            (
                20,
                140
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (
                0,
                0,
                255
            ),

            2
        )


    else:

        cv2.putText(
            annotated,

            "Event Buffer: Last 5 seconds",

            (
                20,
                140
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (
                255,
                255,
                255
            ),

            2
        )


    # =====================================================
    # 25. Show
    # =====================================================

    cv2.imshow(
        "V6 Violence Detection + Event Recorder",
        annotated
    )


    if (
        cv2.waitKey(1)
        & 0xFF
        ==
        ord("q")
    ):

        break


# =========================================================
# 26. 如果錄影途中按 Q
# =========================================================

if (
    recording_event
    and
    len(event_video_frames) >= 2
):

    print()
    print(
        "程式關閉前，儲存目前已取得的事件影片..."
    )


    video_path = save_event_video(
        event_video_frames,
        event_video_times
    )


    if video_path is not None:

        upload_event_video(
            video_path
        )


# =========================================================
# 27. Close
# =========================================================

cap.release()

cv2.destroyAllWindows()


print()
print("=" * 70)
print("V6 Violence Event Recorder Closed")
print("=" * 70)