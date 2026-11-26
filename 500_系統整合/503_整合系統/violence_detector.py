
import numpy as np
import tensorflow as tf

from collections import deque

from config import (
    VIOLENCE_MODEL_FILE,
    SEQUENCE_LENGTH,
    KEYPOINT_CONF_THRESHOLD,
    VIOLENCE_THRESHOLD,
    SELECTED_POINTS,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    MIN_BODY_SCALE,
    CLIP_MIN,
    CLIP_MAX
)


class ViolenceDetector:

    # =====================================================
    # 1. 初始化
    # =====================================================

    def __init__(self):

        print("Loading Violence V6 Model...")

        self.model = tf.keras.models.load_model(
            VIOLENCE_MODEL_FILE,
            compile=False
        )

        self.pose_buffer = deque(
            maxlen=SEQUENCE_LENGTH
        )

        self.motion_buffer = deque(
            maxlen=SEQUENCE_LENGTH
        )

        self.previous_pose = None
        self.previous_valid = None

        print("Violence Model Loaded.")

        for tensor in self.model.inputs:
            print(tensor.name, tensor.shape)


    # =====================================================
    # 2. 清空序列
    # =====================================================

    def reset(self):

        self.pose_buffer.clear()
        self.motion_buffer.clear()

        self.previous_pose = None
        self.previous_valid = None


    # =====================================================
    # 3. 計算人體中心
    # =====================================================

    def get_body_center(self, points, valid):

        if valid[LEFT_HIP] and valid[RIGHT_HIP]:

            return (
                points[LEFT_HIP]
                + points[RIGHT_HIP]
            ) / 2.0

        if (
            valid[LEFT_SHOULDER]
            and valid[RIGHT_SHOULDER]
        ):

            return (
                points[LEFT_SHOULDER]
                + points[RIGHT_SHOULDER]
            ) / 2.0

        return np.mean(
            points[valid],
            axis=0
        )


    # =====================================================
    # 4. 計算人體 Scale
    # =====================================================

    def get_body_scale(self, points, valid, center):

        if (
            valid[LEFT_SHOULDER]
            and valid[RIGHT_SHOULDER]
        ):

            shoulder_width = np.linalg.norm(
                points[LEFT_SHOULDER]
                - points[RIGHT_SHOULDER]
            )

            if shoulder_width >= MIN_BODY_SCALE:
                return float(shoulder_width)

        if valid[LEFT_HIP] and valid[RIGHT_HIP]:

            hip_width = np.linalg.norm(
                points[LEFT_HIP]
                - points[RIGHT_HIP]
            )

            if hip_width >= MIN_BODY_SCALE:
                return float(hip_width)

        valid_points = points[valid]

        if len(valid_points) >= 2:

            min_xy = np.min(
                valid_points,
                axis=0
            )

            max_xy = np.max(
                valid_points,
                axis=0
            )

            bbox_scale = float(
                np.max(max_xy - min_xy)
            )

            if bbox_scale >= MIN_BODY_SCALE:
                return bbox_scale

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


    # =====================================================
    # 5. Pose Normalization
    # =====================================================

    def normalize_pose(self, raw_pose):

        points = np.asarray(
            raw_pose,
            dtype=np.float32
        ).reshape(13, 2)

        valid = ~np.all(
            points == 0,
            axis=1
        )

        normalized = np.zeros(
            (13, 2),
            dtype=np.float32
        )

        if not np.any(valid):

            return normalized.reshape(26), valid

        center = self.get_body_center(
            points,
            valid
        )

        scale = self.get_body_scale(
            points,
            valid,
            center
        )

        normalized[valid] = (
            points[valid] - center
        ) / scale

        normalized[valid] = np.clip(
            normalized[valid],
            CLIP_MIN,
            CLIP_MAX
        )

        normalized[~valid] = 0.0

        return normalized.reshape(26), valid


    # =====================================================
    # 6. Motion Calculation
    # =====================================================

    def calculate_motion(
        self,
        current_pose,
        current_valid
    ):

        motion = np.zeros(
            (13, 2),
            dtype=np.float32
        )

        if (
            self.previous_pose is None
            or self.previous_valid is None
        ):

            return motion.reshape(26)

        current_points = current_pose.reshape(
            13, 2
        )

        previous_points = self.previous_pose.reshape(
            13, 2
        )

        valid_both = (
            current_valid
            & self.previous_valid
        )

        motion[valid_both] = (
            current_points[valid_both]
            - previous_points[valid_both]
        )

        return motion.reshape(26)


    # =====================================================
    # 7. 選擇主要人物
    # =====================================================

    def get_person_area(self, person):

        valid_points = []

        for kp_index in SELECTED_POINTS:

            x, y, conf = person[kp_index]

            if conf >= KEYPOINT_CONF_THRESHOLD:

                valid_points.append([x, y])

        if len(valid_points) < 2:
            return 0.0

        valid_points = np.asarray(
            valid_points,
            dtype=np.float32
        )

        min_xy = np.min(valid_points, axis=0)
        max_xy = np.max(valid_points, axis=0)

        size = max_xy - min_xy

        return float(size[0] * size[1])


    def select_person(self, people):

        selected_person = None
        largest_area = 0.0

        for person in people:

            area = self.get_person_area(person)

            if area > largest_area:

                largest_area = area
                selected_person = person

        return selected_person


    # =====================================================
    # 8. 擷取 13 個關鍵點
    # =====================================================

    def extract_raw_pose(self, person):

        raw_pose = []

        for kp_index in SELECTED_POINTS:

            x, y, conf = person[kp_index]

            if conf >= KEYPOINT_CONF_THRESHOLD:

                raw_pose.extend([
                    float(x),
                    float(y)
                ])

            else:

                raw_pose.extend([
                    0.0,
                    0.0
                ])

        return np.asarray(
            raw_pose,
            dtype=np.float32
        )


    # =====================================================
    # 9. 暴力辨識
    # =====================================================

    def update(self, people):

        # 沒有偵測到人物
        if people is None or len(people) == 0:

            self.reset()

            return "NO_PERSON", None

        selected_person = self.select_person(
            people
        )

        if selected_person is None:

            self.reset()

            return "NO_PERSON", None

        raw_pose = self.extract_raw_pose(
            selected_person
        )

        normalized_pose, valid_mask = (
            self.normalize_pose(raw_pose)
        )

        if not np.any(valid_mask):

            self.reset()

            return "NO_PERSON", None

        motion = self.calculate_motion(
            normalized_pose,
            valid_mask
        )

        self.pose_buffer.append(
            normalized_pose.copy()
        )

        self.motion_buffer.append(
            motion.copy()
        )

        self.previous_pose = (
            normalized_pose.copy()
        )

        self.previous_valid = (
            valid_mask.copy()
        )

        # 尚未收集滿 30 frames
        if len(self.pose_buffer) < SEQUENCE_LENGTH:

            return "COLLECTING", None

        pose_input = np.array(
            self.pose_buffer,
            dtype=np.float32
        )[None, :, :]

        motion_input = np.array(
            self.motion_buffer,
            dtype=np.float32
        )[None, :, :]

        # 每個視窗第一幀 Motion 歸零
        motion_input[0, 0, :] = 0.0

        prediction = self.model.predict(
            [
                pose_input,
                motion_input
            ],
            verbose=0
        )

        probability = float(
            prediction[0][0]
        )

        if probability >= VIOLENCE_THRESHOLD:

            status = "VIOLENCE"

        else:

            status = "NON_VIOLENCE"

        return status, probability


    # =====================================================
    # 10. 取得目前 Buffer 長度
    # =====================================================

    def get_buffer_length(self):

        return len(self.pose_buffer)


# =========================================================
# 11. 獨立測試模型是否能載入
# =========================================================

if __name__ == "__main__":

    detector = ViolenceDetector()

    print()
    print("Violence Detector Ready.")
