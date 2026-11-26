
import numpy as np
import tensorflow as tf

from collections import deque

from config import (
    FALL_MODEL_FILE,
    SEQUENCE_LENGTH,
    KEYPOINT_CONF_THRESHOLD,
    FALL_THRESHOLD,
    SELECTED_POINTS,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP
)


class FallDetector:

    # =====================================================
    # 1. 初始化
    # =====================================================

    def __init__(self):

        print("Loading Fall LSTM Model...")

        self.model = tf.keras.models.load_model(
            FALL_MODEL_FILE,
            compile=False
        )

        self.sequence_buffer = deque(
            maxlen=SEQUENCE_LENGTH
        )

        self.previous_pose = None

        print("Fall Model Loaded.")

        for tensor in self.model.inputs:
            print(tensor.name, tensor.shape)


    # =====================================================
    # 2. 清空序列
    # =====================================================

    def reset(self):

        self.sequence_buffer.clear()

        self.previous_pose = None


    # =====================================================
    # 3. 擷取 13 個人體關鍵點
    # =====================================================

    def extract_raw_pose(self, person):

        pose_data = []

        for point_index, idx in enumerate(
            SELECTED_POINTS
        ):

            x, y, conf = person[idx]

            # 信心值不足
            if conf < KEYPOINT_CONF_THRESHOLD:

                # 使用上一幀的位置
                if self.previous_pose is not None:

                    pose_data.append(
                        self.previous_pose[
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

                pose_data.append(
                    np.array(
                        [x, y],
                        dtype=np.float32
                    )
                )

        return np.asarray(
            pose_data,
            dtype=np.float32
        )


    # =====================================================
    # 4. Pose Normalization
    # =====================================================

    def normalize_pose(self, pose):

        left_hip = pose[LEFT_HIP]

        right_hip = pose[RIGHT_HIP]

        # Hip 無效
        if (
            -1 in left_hip
            or -1 in right_hip
        ):

            return None

        # Hip Center
        center = (
            left_hip + right_hip
        ) / 2.0

        normalized_pose = (
            pose - center
        )

        # Shoulder Distance
        left_shoulder = (
            normalized_pose[LEFT_SHOULDER]
        )

        right_shoulder = (
            normalized_pose[RIGHT_SHOULDER]
        )

        shoulder_distance = np.linalg.norm(
            left_shoulder - right_shoulder
        )

        # 避免除以零
        if shoulder_distance < 1e-6:

            return None

        # Scale Normalization
        normalized_pose = (
            normalized_pose
            / shoulder_distance
        )

        return (
            normalized_pose
            .reshape(26)
            .astype(np.float32)
        )


    # =====================================================
    # 5. 跌倒辨識
    # =====================================================

    def update(self, people):

        # 沒有偵測到人體
        if people is None or len(people) == 0:

            self.reset()

            return "NO_PERSON", None

        # 保留同學原本的第一人選擇方式
        selected_person = people[0]

        # 所有選定關鍵點皆不可靠
        if not np.any(
            selected_person[
                SELECTED_POINTS, 2
            ] >= KEYPOINT_CONF_THRESHOLD
        ):

            self.reset()

            return "NO_PERSON", None

        # 擷取骨架
        raw_pose = self.extract_raw_pose(
            selected_person
        )

        # 儲存上一幀
        self.previous_pose = raw_pose.copy()

        # 正規化
        normalized_pose = self.normalize_pose(
            raw_pose
        )

        if normalized_pose is None:

            self.sequence_buffer.clear()

            return "COLLECTING", None

        # 加入 Buffer
        self.sequence_buffer.append(
            normalized_pose.copy()
        )

        # 尚未收集滿 30 Frames
        if (
            len(self.sequence_buffer)
            < SEQUENCE_LENGTH
        ):

            return "COLLECTING", None

        # 建立模型輸入
        sequence = np.asarray(
            self.sequence_buffer,
            dtype=np.float32
        )

        sequence = np.expand_dims(
            sequence,
            axis=0
        )

        # LSTM 預測
        prediction = self.model.predict(
            sequence,
            verbose=0
        )

        probability = float(
            prediction[0][0]
        )

        # 判斷是否跌倒
        if probability >= FALL_THRESHOLD:

            status = "FALL"

        else:

            status = "NON_FALL"

        return status, probability


    # =====================================================
    # 6. 取得 Buffer 長度
    # =====================================================

    def get_buffer_length(self):

        return len(self.sequence_buffer)


# =========================================================
# 7. 獨立測試模型載入
# =========================================================

if __name__ == "__main__":

    detector = FallDetector()

    print()
    print("Fall Detector Ready.")
