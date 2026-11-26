import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from ultralytics import YOLO

# 1. 載入 27 種真值表查表模組
from rule_table import get_final_decision

# 2. 載入自定義 PPG 模型類別
from models.ppg_model import StressNN

# 3. 載入異常監控與通報模組
from anomaly_monitor import AnomalyDetector, send_alert_notification


# ----------------------------------------------------
# 1. 預設 Placeholder 模型 (當組員尚未提供模型時使用)
# ----------------------------------------------------
class DummyEDAModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super(DummyEDAModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.classes = ["Normal", "Suspected_Arousal", "High_Arousal"]

    def forward(self, x):
        return self.net(x)

    def predict_state(self, eda_tensor):
        """推論並返回 EDA 三階段狀態"""
        with torch.no_grad():
            logits = self.forward(eda_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
        return self.classes[pred_idx]


class DummyIMUModel(nn.Module):
    def __init__(self, in_channels=6, num_classes=3):
        super(DummyIMUModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.classes = ["calm", "slightly_intense", "fierce"]

    def forward(self, x):
        return self.net(x)

    def predict_state(self, imu_raw, imu_tensor):
        """優先以物理 G 值加速度劃分 IMU 狀態，兼顧模型輸出"""
        acc_x, acc_y, acc_z = imu_raw[0, 0, :], imu_raw[0, 1, :], imu_raw[0, 2, :]
        max_g = np.max(np.sqrt(acc_x**2 + acc_y**2 + acc_z**2))

        if max_g < 1.2:
            return "calm"
        elif 1.2 <= max_g <= 2.5:
            return "slightly_intense"
        else:
            return "fierce"


# ----------------------------------------------------
# 2. 多模態壓力與異常檢測預測器類別 (PPG, EDA, IMU)
# ----------------------------------------------------
class MultimodalStressPredictor:
    """
    多模態 (PPG, EDA, IMU) 27 種真值表查表推論整合類別
    """

    def __init__(self, weights_dir="weights"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = weights_dir

        # 載入 PPG 相關權重與預處理器
        ppg_scaler_path = os.path.join(weights_dir, "ppg_scaler.pkl")
        label_encoder_path = os.path.join(weights_dir, "label_encoder.pkl")
        ppg_weight_path = os.path.join(weights_dir, "ppg_stress_model.pth")

        if os.path.exists(ppg_scaler_path) and os.path.exists(label_encoder_path):
            self.ppg_scaler = joblib.load(ppg_scaler_path)
            self.label_encoder = joblib.load(label_encoder_path)
            self.num_classes = len(self.label_encoder.classes_)
            print("✓ 成功載入 ppg_scaler.pkl 與 label_encoder.pkl")
        else:
            print("⚠️ 未找到 ppg_scaler.pkl 或 label_encoder.pkl，推論時將跳過 Scaler 標準化。")
            self.ppg_scaler = None
            self.label_encoder = None
            self.num_classes = 3

        # 初始化 PPG 模型
        self.ppg_model = StressNN(input_dim=19, num_classes=self.num_classes).to(self.device)
        if os.path.exists(ppg_weight_path):
            self.ppg_model.load_state_dict(
                torch.load(ppg_weight_path, map_location=self.device)
            )
            print("✓ 成功載入 ppg_stress_model.pth 權重")
        else:
            print("⚠️ 未找到 ppg_stress_model.pth，使用未訓練的預設 PPG 權重進行測試。")
        self.ppg_model.eval()

        # 載入 EDA 模型
        self.eda_model = DummyEDAModel(
            in_channels=1, num_classes=self.num_classes
        ).to(self.device)
        eda_weight_path = os.path.join(weights_dir, "eda_model.pth")
        if os.path.exists(eda_weight_path):
            self.eda_model.load_state_dict(
                torch.load(eda_weight_path, map_location=self.device)
            )
            print("✓ 成功載入 eda_model.pth 權重")
        else:
            print("ℹ️ 未找到 eda_model.pth，使用 Dummy EDA 模型。")
        self.eda_model.eval()

        # 載入 IMU 模型
        self.imu_model = DummyIMUModel(
            in_channels=6, num_classes=self.num_classes
        ).to(self.device)
        imu_weight_path = os.path.join(weights_dir, "imu_model.pth")
        if os.path.exists(imu_weight_path):
            self.imu_model.load_state_dict(
                torch.load(imu_weight_path, map_location=self.device)
            )
            print("✓ 成功載入 imu_model.pth 權重")
        else:
            print("ℹ️ 未找到 imu_model.pth，使用 Dummy IMU 模型。")
        self.imu_model.eval()

    def get_ppg_state(self, ppg_raw):
        """將 PPG 的心率 (HR) 數值離散化為 Slow / Medium / Fast"""
        hr = ppg_raw[0][6]  # Index 6 為 HR 心率
        if hr < 60:
            return "Slow"
        elif 60 <= hr <= 100:
            return "Medium"
        else:
            return "Fast"

    def predict(self, ppg_raw, eda_raw, imu_raw):
        """
        三模態狀態映射與 27 種真值表查表推論
        """
        eda_tensor = torch.tensor(eda_raw, dtype=torch.float32).to(self.device)
        imu_tensor = torch.tensor(imu_raw, dtype=torch.float32).to(self.device)

        ppg_state = self.get_ppg_state(ppg_raw)
        eda_state = self.eda_model.predict_state(eda_tensor)
        imu_state = self.imu_model.predict_state(imu_raw, imu_tensor)

        final_decision = get_final_decision(ppg_state, eda_state, imu_state)

        return {
            "ppg_state": ppg_state,
            "eda_state": eda_state,
            "imu_state": imu_state,
            "final_decision": final_decision,
        }


# ----------------------------------------------------
# 3. 合併影像雙任務偵測器 (單一 Camera + 單一 YOLO Pose)
#    同時執行：暴力判斷 (V6 Dual-Branch LSTM) + 跌倒判斷 (Fall LSTM)
# ----------------------------------------------------
class DualVisionAnomalyDetector:
    # 13 個骨架點在 YOLO11 Pose 17 點中的索引
    SELECTED_POINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    # 13 點陣列中的內部索引
    LEFT_SHOULDER = 1
    RIGHT_SHOULDER = 2
    LEFT_HIP = 7
    RIGHT_HIP = 8

    # 暴力正規化常數 (與 402_realtime_violence_event_recorder.py 一致)
    MIN_BODY_SCALE = 5.0
    CLIP_MIN = -20.0
    CLIP_MAX = 20.0

    def __init__(
        self,
        pose_model_path="weights/yolo11n-pose.pt",
        violence_model_path="weights/lstm_v6_dual_branch.keras",
        fall_model_path="weights/fall_lstm_model.keras",
        save_dir="event_videos",
        seq_len=30,
        conf_thresh=0.5,
        violence_thresh=0.5,
        fall_thresh=0.7,
    ):
        self.seq_len = seq_len
        self.conf_thresh = conf_thresh
        self.violence_thresh = violence_thresh
        self.fall_thresh = fall_thresh

        self.save_folder = Path(save_dir)
        self.save_folder.mkdir(parents=True, exist_ok=True)

        # 1. 載入共用 YOLO11 Pose 與兩個獨立 LSTM 模型
        print("Loading YOLO11 Pose model...")
        if not Path(pose_model_path).exists():
            raise FileNotFoundError(f"找不到 YOLO Pose 模型：{pose_model_path}")
        self.pose_model = YOLO(pose_model_path)

        print("Loading V6 Dual-Branch LSTM (Violence)...")
        if not Path(violence_model_path).exists():
            raise FileNotFoundError(f"找不到暴力偵測模型：{violence_model_path}")
        self.violence_model = tf.keras.models.load_model(violence_model_path)

        print("Loading Fall LSTM model...")
        if not Path(fall_model_path).exists():
            raise FileNotFoundError(f"找不到跌倒偵測模型：{fall_model_path}")
        self.fall_model = tf.keras.models.load_model(fall_model_path)

        print("✓ 影像模型全部載入完成。")

        # 2. 暴力偵測專用 Buffer 與前一幀狀態
        self.v_pose_buf = deque(maxlen=seq_len)
        self.v_motion_buf = deque(maxlen=seq_len)
        self.v_prev_norm_pose = None
        self.v_prev_valid_mask = None

        # 3. 跌倒偵測專用 Buffer 與前一幀原始座標
        self.f_pose_buf = deque(maxlen=seq_len)
        self.f_prev_raw_pose = None

        # 4. 共用 5 秒影片緩衝與錄影狀態
        self.pre_event_seconds = 5.0
        self.post_event_seconds = 5.0
        self.cooldown_seconds = 10.0

        self.video_buffer = deque()
        self.recording_event = False
        self.recording_type = ""  # "VIOLENCE" 或 "FALL"
        self.event_video_frames = []
        self.event_video_times = []
        self.recording_end_time = 0.0
        self.last_video_saved_time = -self.cooldown_seconds

        self.prev_v_status = "NON_VIOLENCE"
        self.prev_f_status = "NON_FALL"

    # -----------------------------------------------------
    # 找畫面中面積最大的主要人物
    # -----------------------------------------------------
    def _select_main_person(self, people):
        selected_person = None
        largest_area = 0.0

        for person in people:
            valid_points = []
            for kp_index in self.SELECTED_POINTS:
                x, y, conf = person[kp_index]
                if conf >= self.conf_thresh:
                    valid_points.append([x, y])

            if len(valid_points) < 2:
                continue

            pts_arr = np.array(valid_points, dtype=np.float32)
            min_xy = np.min(pts_arr, axis=0)
            max_xy = np.max(pts_arr, axis=0)
            area = float((max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]))

            if area > largest_area:
                largest_area = area
                selected_person = person

        return selected_person

    # -----------------------------------------------------
    # 暴力模型專用前處理 (保留原 402 腳本的補 0、多層級 Scale 與 Motion)
    # -----------------------------------------------------
    def _update_violence_buffers(self, person):
        raw_pose = []
        for kp_index in self.SELECTED_POINTS:
            x, y, conf = person[kp_index]
            if conf >= self.conf_thresh:
                raw_pose.extend([float(x), float(y)])
            else:
                raw_pose.extend([0.0, 0.0])

        points = np.array(raw_pose, dtype=np.float32).reshape(13, 2)
        valid = ~np.all(points == 0, axis=1)

        if not np.any(valid):
            norm_pose = np.zeros(26, dtype=np.float32)
        else:
            # 1. 計算人體中心點
            if valid[self.LEFT_HIP] and valid[self.RIGHT_HIP]:
                center = (points[self.LEFT_HIP] + points[self.RIGHT_HIP]) / 2.0
            elif valid[self.LEFT_SHOULDER] and valid[self.RIGHT_SHOULDER]:
                center = (points[self.LEFT_SHOULDER] + points[self.RIGHT_SHOULDER]) / 2.0
            else:
                center = np.mean(points[valid], axis=0)

            # 2. 計算人體 Scale
            scale = self.MIN_BODY_SCALE
            if valid[self.LEFT_SHOULDER] and valid[self.RIGHT_SHOULDER]:
                shoulder_w = float(
                    np.linalg.norm(points[self.LEFT_SHOULDER] - points[self.RIGHT_SHOULDER])
                )
                if shoulder_w >= self.MIN_BODY_SCALE:
                    scale = shoulder_w
            elif valid[self.LEFT_HIP] and valid[self.RIGHT_HIP]:
                hip_w = float(
                    np.linalg.norm(points[self.LEFT_HIP] - points[self.RIGHT_HIP])
                )
                if hip_w >= self.MIN_BODY_SCALE:
                    scale = hip_w
            elif np.sum(valid) >= 2:
                vp = points[valid]
                bbox_scale = float(np.max(np.max(vp, axis=0) - np.min(vp, axis=0)))
                if bbox_scale >= self.MIN_BODY_SCALE:
                    scale = bbox_scale
            else:
                distances = np.linalg.norm(points[valid] - center, axis=1)
                if len(distances) > 0 and float(np.max(distances)) >= self.MIN_BODY_SCALE:
                    scale = float(np.max(distances))

            normalized = np.zeros_like(points, dtype=np.float32)
            normalized[valid] = np.clip(
                (points[valid] - center) / scale, self.CLIP_MIN, self.CLIP_MAX
            )
            norm_pose = normalized.reshape(26)

        # 3. 計算 Motion
        if self.v_prev_norm_pose is None or self.v_prev_valid_mask is None:
            motion = np.zeros(26, dtype=np.float32)
        else:
            curr_pts = norm_pose.reshape(13, 2)
            prev_pts = self.v_prev_norm_pose.reshape(13, 2)
            m_pts = np.zeros((13, 2), dtype=np.float32)
            valid_both = valid & self.v_prev_valid_mask
            m_pts[valid_both] = curr_pts[valid_both] - prev_pts[valid_both]
            motion = m_pts.reshape(26)

        self.v_pose_buf.append(norm_pose)
        self.v_motion_buf.append(motion)
        self.v_prev_norm_pose = norm_pose.copy()
        self.v_prev_valid_mask = valid.copy()

    # -----------------------------------------------------
    # 跌倒模型專用前處理 (保留原 realtime_fall_detection 補上一幀、髖中心、肩寬縮放)
    # -----------------------------------------------------
    def _update_fall_buffers(self, person):
        pose_data = []
        for point_index, idx in enumerate(self.SELECTED_POINTS):
            x, y, conf = person[idx]
            if conf < self.conf_thresh:
                if self.f_prev_raw_pose is not None:
                    pose_data.append(self.f_prev_raw_pose[point_index].copy())
                else:
                    pose_data.append(np.array([-1.0, -1.0], dtype=np.float32))
            else:
                pose_data.append(np.array([float(x), float(y)], dtype=np.float32))

        pose_arr = np.array(pose_data, dtype=np.float32)
        self.f_prev_raw_pose = pose_arr.copy()

        left_hip = pose_arr[self.LEFT_HIP]
        right_hip = pose_arr[self.RIGHT_HIP]
        if -1 in left_hip or -1 in right_hip:
            return

        center = (left_hip + right_hip) / 2.0
        norm_pose = pose_arr - center

        shoulder_distance = np.linalg.norm(
            norm_pose[self.LEFT_SHOULDER] - norm_pose[self.RIGHT_SHOULDER]
        )
        if shoulder_distance < 1e-6:
            return

        norm_pose = (norm_pose / shoulder_distance).reshape(26).astype(np.float32)
        self.f_pose_buf.append(norm_pose)

    # -----------------------------------------------------
    # 儲存異常事件影片 (前 5 秒 + 後 5 秒)
    # -----------------------------------------------------
    def save_event_video(self, frames, frame_times, event_prefix):
        if len(frames) < 2:
            print("影片畫面不足，無法儲存。")
            return None

        video_duration = frame_times[-1] - frame_times[0]
        actual_fps = (len(frames) - 1) / max(video_duration, 0.001)
        actual_fps = max(1.0, min(actual_fps, 60.0))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = self.save_folder / f"{event_prefix.lower()}_{timestamp}.mp4"
        frame_height, frame_width = frames[0].shape[:2]

        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            actual_fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            print("影片建立失敗。")
            return None

        for saved_frame in frames:
            writer.write(saved_frame)
        writer.release()

        print("=" * 60)
        print(f"異常事件影片儲存完成 [{event_prefix}]")
        print(f"影片路徑：{video_path}")
        print(f"實際長度：約 {video_duration:.2f} 秒 | FPS：約 {actual_fps:.2f}")
        print("=" * 60)
        return str(video_path)

    # -----------------------------------------------------
    # 單幀推論主函式
    # -----------------------------------------------------
    def process_frame(self, frame):
        current_time = time.monotonic()

        # 1. 只執行一次 YOLO11 Pose
        results = self.pose_model(frame, verbose=False)
        result = results[0]
        annotated = result.plot()

        v_status = "COLLECTING"
        v_prob = 0.0
        f_status = "COLLECTING"
        f_prob = 0.0

        # 2. 提取主要人物並分別更新暴力與跌倒特徵 Buffer
        if (
            result.keypoints is not None
            and result.keypoints.data is not None
            and len(result.keypoints.data) > 0
        ):
            people = result.keypoints.data.cpu().numpy()
            main_person = self._select_main_person(people)
            if main_person is not None:
                self._update_violence_buffers(main_person)
                self._update_fall_buffers(main_person)

        # 3. 暴力模型推論 (滿 30 幀)
        if (
            len(self.v_pose_buf) == self.seq_len
            and len(self.v_motion_buf) == self.seq_len
        ):
            pose_input = np.expand_dims(
                np.array(self.v_pose_buf, dtype=np.float32), axis=0
            )
            motion_input = np.expand_dims(
                np.array(self.v_motion_buf, dtype=np.float32), axis=0
            )
            v_pred = self.violence_model.predict([pose_input, motion_input], verbose=0)
            v_prob = float(v_pred[0][0])
            v_status = "VIOLENCE" if v_prob >= self.violence_thresh else "NON_VIOLENCE"

        # 4. 跌倒模型推論 (滿 30 幀)
        if len(self.f_pose_buf) == self.seq_len:
            fall_input = np.expand_dims(
                np.array(self.f_pose_buf, dtype=np.float32), axis=0
            )
            f_pred = self.fall_model.predict(fall_input, verbose=0)
            f_prob = float(f_pred[0][0])
            f_status = "FALL" if f_prob >= self.fall_thresh else "NON_FALL"

        # 5. 在畫面上同時繪製暴力與跌倒辨識結果
        v_color = (
            (0, 0, 255)
            if v_status == "VIOLENCE"
            else ((0, 255, 0) if v_status == "NON_VIOLENCE" else (0, 255, 255))
        )
        f_color = (
            (0, 0, 255)
            if f_status == "FALL"
            else ((0, 255, 0) if f_status == "NON_FALL" else (0, 255, 255))
        )

        cv2.putText(
            annotated,
            f"Violence: {v_status} ({v_prob:.3f}) [{len(self.v_pose_buf)}/{self.seq_len}]",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            v_color,
            2,
        )
        cv2.putText(
            annotated,
            f"Fall:     {f_status} ({f_prob:.3f}) [{len(self.f_pose_buf)}/{self.seq_len}]",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            f_color,
            2,
        )

        # 6. 共用前後 5 秒緩衝與事件觸發錄影
        current_video_frame = annotated.copy()

        if not self.recording_event:
            self.video_buffer.append((current_time, current_video_frame))
            while (
                self.video_buffer
                and (current_time - self.video_buffer[0][0]) > self.pre_event_seconds
            ):
                self.video_buffer.popleft()

        new_violence_event = (
            v_status == "VIOLENCE" and self.prev_v_status != "VIOLENCE"
        )
        new_fall_event = f_status == "FALL" and self.prev_f_status != "FALL"
        cooldown_finished = (
            current_time - self.last_video_saved_time
        ) >= self.cooldown_seconds

        if (
            (new_violence_event or new_fall_event)
            and not self.recording_event
            and cooldown_finished
        ):
            self.recording_event = True
            self.recording_type = "VIOLENCE" if new_violence_event else "FALL"

            self.event_video_times = [t for t, _ in self.video_buffer]
            self.event_video_frames = [f.copy() for _, f in self.video_buffer]
            self.recording_end_time = current_time + self.post_event_seconds

            print("=" * 60)
            print(f"⚠️ 偵測到 {self.recording_type}！正在保存異常前 5 秒 + 後 5 秒...")
            print("=" * 60)

        saved_video_path = None
        if self.recording_event:
            if (
                not self.event_video_times
                or current_time > self.event_video_times[-1]
            ):
                self.event_video_times.append(current_time)
                self.event_video_frames.append(current_video_frame.copy())

            remaining = max(0.0, self.recording_end_time - current_time)
            cv2.putText(
                annotated,
                f"EVENT RECORDING ({self.recording_type}): {remaining:.1f}s",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            if current_time >= self.recording_end_time:
                saved_video_path = self.save_event_video(
                    self.event_video_frames,
                    self.event_video_times,
                    self.recording_type,
                )
                self.last_video_saved_time = current_time
                self.recording_event = False
                self.recording_type = ""
                self.event_video_frames.clear()
                self.event_video_times.clear()
                self.video_buffer.clear()
        else:
            cv2.putText(
                annotated,
                "Event Buffer: Last 5 seconds",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        self.prev_v_status = v_status
        self.prev_f_status = f_status

        return annotated, {
            "violence_status": v_status,
            "violence_prob": v_prob,
            "fall_status": f_status,
            "fall_prob": f_prob,
            "saved_video_path": saved_video_path,
        }

    # -----------------------------------------------------
    # 程式提前關閉時，若正在錄影則強制儲存已錄製片段
    # -----------------------------------------------------
    def flush_pending_recording(self):
        if self.recording_event and len(self.event_video_frames) >= 2:
            print("程式關閉前，儲存目前已取得的事件影片...")
            return self.save_event_video(
                self.event_video_frames,
                self.event_video_times,
                self.recording_type,
            )
        return None


# ----------------------------------------------------
# 4. 主執行流程 (單一 Camera 同步執行暴力 + 跌倒偵測)
# ----------------------------------------------------
if __name__ == "__main__":
    print("=== 初始化多模態壓力檢測預測器與異常監控器 ===")
    predictor = MultimodalStressPredictor(weights_dir="weights")
    detector = AnomalyDetector()

    print("\n=== 初始化雙任務影像偵測器 (暴力 + 跌倒) ===")
    vision_detector = DualVisionAnomalyDetector(
        pose_model_path="weights/yolo11n-pose.pt",
        violence_model_path="weights/lstm_v6_dual_branch.keras",
        fall_model_path="weights/fall_lstm_model.keras",
        save_dir="event_videos",
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Camera Open:", cap.isOpened())
    if not cap.isOpened():
        raise RuntimeError("無法開啟 Webcam")

    print("\n============================================================")
    print("系統啟動中：單一攝影機同步進行 [暴力判斷 + 跌倒判斷]")
    print("平時在 RAM 保留最近 5 秒畫面，觸發異常後再錄 5 秒並輸出 MP4")
    print("按 Q 離開")
    print("============================================================")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("讀不到攝影機畫面")
                break

            # 1. 執行單幀雙任務影像推論
            annotated_frame, vision_res = vision_detector.process_frame(frame)

            # 2. 當有異常事件影片儲存完成時，可在此串接通報模組
            if vision_res["saved_video_path"] is not None:
                print(f"異常影片已就緒，可進行通報上傳：{vision_res['saved_video_path']}")
                # 若要結合生理訊號通報，可在此呼叫：
                # res_sensor = predictor.predict(ppg_raw, eda_raw, imu_raw)
                # report = detector.evaluate_status(res_sensor)
                # send_alert_notification(report)

            # 3. 顯示合併後的即時畫面
            cv2.imshow("Unified Vision Detection (Violence + Fall)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # 確保按 Q 離開時，未錄完的異常片段也能存檔並正確釋放攝影機
        pending_video = vision_detector.flush_pending_recording()
        if pending_video is not None:
            print(f"已補存中斷前影片：{pending_video}")

        cap.release()
        cv2.destroyAllWindows()
        print("=== 系統已安全關閉 ===")
