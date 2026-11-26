from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from datetime import datetime
import os
import time


# ==========================================
# 1. 基本設定
# ==========================================

POSE_MODEL_FILE = "yolo11n-pose.pt"
LSTM_MODEL_FILE = "fall_lstm_model.keras"

SEQUENCE_LENGTH = 30

# LSTM 輸出 >= 0.7 判定為跌倒
FALL_THRESHOLD = 0.7

# YOLO Keypoint 信心值門檻
KEYPOINT_CONF_THRESHOLD = 0.5


# ==========================================
# 2. 跌倒影片設定
# ==========================================

VIDEO_SAVE_FOLDER = "fall_videos"

# 跌倒前 5 秒＋跌倒後 5 秒
PRE_FALL_SECONDS = 5
POST_FALL_SECONDS = 5

# 儲存影片後，至少間隔 10 秒才允許再次觸發
RECORD_COOLDOWN_SECONDS = 10

os.makedirs(
    VIDEO_SAVE_FOLDER,
    exist_ok=True
)


# ==========================================
# 3. 載入模型
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
# 4. 保留的 13 個骨架點
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
# 5. 建立 30 Frame Buffer
# ==========================================

sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)


# ==========================================
# 6. 儲存上一幀有效骨架
# ==========================================

previous_pose = None


# ==========================================
# 7. 跌倒影片錄影狀態
# ==========================================

# 保存最近 5 秒的畫面
# 每個元素格式：(時間, 畫面)
video_buffer = deque()

# 是否正在錄製跌倒影片
recording_fall = False

# 實際準備寫入影片的畫面
fall_video_frames = []

# 每張畫面的取得時間
fall_video_times = []

# 跌倒後錄影的結束時間
recording_end_time = 0.0

# 上一次影片儲存完成的時間
last_video_saved_time = -RECORD_COOLDOWN_SECONDS

# 上一幀的辨識狀態
previous_status = "NON_FALL"


# ==========================================
# 8. 儲存影片函式
# ==========================================

def save_fall_video(frames, frame_times):

    """
    把跌倒前後的畫面儲存成 MP4。
    """

    if len(frames) < 2:

        print("影片畫面不足，無法儲存。")

        return None


    # --------------------------------------
    # 根據實際處理速度計算 FPS
    # --------------------------------------

    video_duration = (
        frame_times[-1]
        -
        frame_times[0]
    )

    actual_fps = (
        (len(frames) - 1)
        /
        max(video_duration, 0.001)
    )

    # 防止 FPS 太低或太高
    actual_fps = max(
        1.0,
        min(actual_fps, 60.0)
    )


    # --------------------------------------
    # 建立檔案名稱
    # --------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    video_filename = os.path.join(
        VIDEO_SAVE_FOLDER,
        f"fall_{timestamp}.mp4"
    )


    # --------------------------------------
    # 取得影片大小
    # --------------------------------------

    frame_height, frame_width = (
        frames[0].shape[:2]
    )


    # --------------------------------------
    # 建立影片
    # --------------------------------------

    video_writer = cv2.VideoWriter(
        video_filename,

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        actual_fps,

        (
            frame_width,
            frame_height
        )
    )


    if not video_writer.isOpened():

        print(
            "影片建立失敗，請檢查編碼格式或資料夾權限。"
        )

        return None


    # --------------------------------------
    # 寫入所有畫面
    # --------------------------------------

    for saved_frame in frames:

        video_writer.write(
            saved_frame
        )


    video_writer.release()

    print(
        f"跌倒影片已儲存：{video_filename}"
    )

    print(
        f"影片長度：約 {video_duration:.1f} 秒"
    )

    print(
        f"影片 FPS：{actual_fps:.1f}"
    )

    return video_filename


# ==========================================
# 9. Pose 正規化函式
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
# 10. 開啟攝影機
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
# 11. 即時辨識
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
                    # 如果這一點抓不到，
                    # 就使用上一幀的位置
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
    # Buffer 滿 30 Frames 才進行預測
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


        # 增加 Batch Dimension
        #
        # (30, 26)
        # 變成
        # (1, 30, 26)

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
    # 12. 顯示 Buffer 狀態
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
    # 13. 設定文字顏色
    # ======================================

    if status == "FALL":

        text_color = (
            0,
            0,
            255
        )

    elif status == "NON_FALL":

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


    # ======================================
    # 14. 顯示辨識結果
    # ======================================

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
    # 15. 顯示跌倒機率
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
    # 16. 跌倒影片處理
    # ======================================

    current_time = time.monotonic()

    # 儲存有骨架、狀態和機率的畫面
    current_video_frame = annotated.copy()


    # --------------------------------------
    # 沒有正式錄影時：
    # 保存最近 5 秒的畫面
    # --------------------------------------

    if not recording_fall:

        video_buffer.append(
            (
                current_time,
                current_video_frame
            )
        )


        # 刪除超過 5 秒的舊畫面
        while (
            video_buffer
            and
            current_time
            -
            video_buffer[0][0]
            >
            PRE_FALL_SECONDS
        ):

            video_buffer.popleft()


    # --------------------------------------
    # 判斷是不是新發生的跌倒
    # --------------------------------------

    new_fall_event = (
        status == "FALL"
        and
        previous_status != "FALL"
    )


    cooldown_finished = (
        current_time
        -
        last_video_saved_time
        >=
        RECORD_COOLDOWN_SECONDS
    )


    # --------------------------------------
    # 開始建立跌倒影片
    # --------------------------------------

    if (
        new_fall_event
        and
        not recording_fall
        and
        cooldown_finished
    ):

        recording_fall = True


        # 複製跌倒前 5 秒的時間
        fall_video_times = [
            saved_time
            for saved_time, saved_frame
            in video_buffer
        ]


        # 複製跌倒前 5 秒的畫面
        fall_video_frames = [
            saved_frame.copy()
            for saved_time, saved_frame
            in video_buffer
        ]


        # 設定跌倒後錄影的結束時間
        recording_end_time = (
            current_time
            +
            POST_FALL_SECONDS
        )


        print(
            "偵測到跌倒！"
        )

        print(
            "正在保存跌倒前 5 秒與跌倒後 5 秒..."
        )


    # --------------------------------------
    # 正在錄製跌倒後的畫面
    # --------------------------------------

    if recording_fall:

        # 避免觸發當下的畫面重複加入
        if (
            not fall_video_times
            or
            current_time
            >
            fall_video_times[-1]
        ):

            fall_video_times.append(
                current_time
            )

            fall_video_frames.append(
                current_video_frame
            )


        # ----------------------------------
        # 跌倒後 5 秒錄製完成
        # ----------------------------------

        if current_time >= recording_end_time:

            save_fall_video(
                fall_video_frames,
                fall_video_times
            )


            # 記錄完成時間
            last_video_saved_time = (
                current_time
            )


            # 重設錄影狀態
            recording_fall = False

            fall_video_frames.clear()
            fall_video_times.clear()
            video_buffer.clear()


    # 更新上一幀狀態
    previous_status = status


    # ======================================
    # 17. 顯示畫面
    # ======================================

    cv2.imshow(
        "Real-Time Fall Detection",
        annotated
    )


    # ======================================
    # 18. 按 q 離開
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
# 19. 如果錄影中按下 q
# 儲存目前已經取得的影片
# ==========================================

if (
    recording_fall
    and
    len(fall_video_frames) >= 2
):

    print(
        "程式關閉前，正在儲存尚未完成的跌倒影片..."
    )

    save_fall_video(
        fall_video_frames,
        fall_video_times
    )


# ==========================================
# 20. 關閉攝影機
# ==========================================

cap.release()

cv2.destroyAllWindows()

print(
    "Real-Time Fall Detection Closed."
)