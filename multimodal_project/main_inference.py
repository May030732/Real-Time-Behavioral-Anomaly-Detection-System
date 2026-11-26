import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import tensorflow as tf
import torch
from ultralytics import YOLO

# 1. 載入 36 種真值表查表模組
from rule_table import get_final_decision

# 2. 載入自定義 PPG 模型類別
from models.ppg_model import StressNN

# 3. 載入隊友的正式 EDA 與 IMU 模組
from eda_detector import EDADetector
import imu_test

# 4. 載入異常監控與通報模組
from anomaly_monitor import AnomalyDetector, send_alert_notification


# ----------------------------------------------------
# 1. 多模態穿戴式感測器預測器 (PPG 4類 + 隊友 EDA + 隊友 IMU)
# ----------------------------------------------------
class MultimodalWearablePredictor:
    VALID_PPG_STATES = {
        "NORMAL",
        "PERSISTENT_HIGH_HR",
        "SUDDEN_HR_CHANGE",
        "CRITICAL_EXTREME_HR",
    }

    def __init__(
        self,
        weights_dir="weights",
        eda_baseline_count=30,
        eda_suspected_thresh=0.20,
        eda_high_thresh=0.50,
        eda_required_count=3,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = weights_dir

        # --- 1. 初始化 PPG 模型 ---
        ppg_scaler_path = os.path.join(weights_dir, "ppg_scaler.pkl")
        label_encoder_path = os.path.join(weights_dir, "label_encoder.pkl")
        ppg_weight_path = os.path.join(weights_dir, "ppg_stress_model.pth")

        if os.path.exists(ppg_scaler_path) and os.path.exists(label_encoder_path):
            self.ppg_scaler = joblib.load(ppg_scaler_path)
            self.label_encoder = joblib.load(label_encoder_path)
            self.num_classes = len(self.label_encoder.classes_)
            print("✓ 成功載入 ppg_scaler.pkl 與 label_encoder.pkl")
        else:
            print("⚠️ 未找到 ppg_scaler.pkl 或 label_encoder.pkl，將使用心率規則映射 PPG 4 類狀態。")
            self.ppg_scaler = None
            self.label_encoder = None
            self.num_classes = 4

        self.ppg_model = StressNN(input_dim=19, num_classes=self.num_classes).to(self.device)
        if os.path.exists(ppg_weight_path):
            self.ppg_model.load_state_dict(
                torch.load(ppg_weight_path, map_location=self.device)
            )
            print("✓ 成功載入 ppg_stress_model.pth 權重")
        self.ppg_model.eval()

        self.prev_hr = None

        # --- 2. 初始化隊友 EDADetector ---
        self.eda_detector = EDADetector(
            baseline_sample_count=eda_baseline_count,
            suspected_threshold=eda_suspected_thresh,
            high_threshold=eda_high_thresh,
            required_abnormal_count=eda_required_count,
        )
        print("✓ 成功初始化隊友 EDADetector")

        # --- 3. 檢查隊友 IMU Random Forest 模型 ---
        imu_info = imu_test.get_model_info()
        print(
            f"✓ 成功載入隊友 IMU 模型：{imu_info['model_file']} "
            f"(Window: {imu_info['window_size']} 筆)"
        )

    def get_ppg_state(self, ppg_raw):
        """
        輸出符合 36 種規則表的 4 種 PPG 狀態：
        NORMAL / PERSISTENT_HIGH_HR / SUDDEN_HR_CHANGE / CRITICAL_EXTREME_HR
        """
        if self.ppg_scaler is not None and self.label_encoder is not None:
            scaled = self.ppg_scaler.transform(ppg_raw)
            tensor_in = torch.tensor(scaled, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits = self.ppg_model(tensor_in)
                pred_idx = torch.argmax(logits, dim=1).item()
                pred_label = str(self.label_encoder.inverse_transform([pred_idx])[0])
            if pred_label in self.VALID_PPG_STATES:
                return pred_label

        hr = float(ppg_raw[0][6])  # Index 6 為 HR 心率
        if hr >= 150 or hr <= 40:
            state = "CRITICAL_EXTREME_HR"
        elif self.prev_hr is not None and abs(hr - self.prev_hr) >= 25:
            state = "SUDDEN_HR_CHANGE"
        elif hr > 100:
            state = "PERSISTENT_HIGH_HR"
        else:
            state = "NORMAL"

        self.prev_hr = hr
        return state

    def predict(self, ppg_raw, current_eda_value, imu_rows):
        """
        執行手環三模態推論並查詢 36 種真值表：
        :param ppg_raw: shape (1, 19) HRV 特徵陣列
        :param current_eda_value: 單筆最新 EDA 數值 (float)
        :param imu_rows: list of [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z] (120 筆)
        """
        ppg_state = self.get_ppg_state(ppg_raw)
        eda_state = self.eda_detector.detect_eda(current_eda_value)
        imu_state = imu_test.predict_imu_rows(imu_rows)

        eda_for_table = (
            eda_state
            if eda_state in {"Normal", "Suspected_Arousal", "High_Arousal"}
            else "Normal"
        )
        final_decision = get_final_decision(ppg_state, eda_for_table, imu_state)

        return {
            "ppg_state": ppg_state,
            "eda_state": eda_state,
            "imu_state": imu_state,
            "final_decision": final_decision,
        }


# ----------------------------------------------------
# 2. 合併影像雙任務偵測器 (單一 Camera + 單一 YOLO Pose)
#    同時執行：暴力判斷 (V6 Dual-Branch LSTM) + 跌倒判斷 (Fall LSTM)
# ----------------------------------------------------
class DualVisionAnomalyDetector:
    SELECTED_POINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    LEFT_SHOULDER = 1
    RIGHT_SHOULDER = 2
    LEFT_HIP = 7
    RIGHT_HIP = 8

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

        # 暴力偵測專用 Buffer
        self.v_pose_buf = deque(maxlen=seq_len)
        self.v_motion_buf = deque(maxlen=seq_len)
        self.v_prev_norm_pose = None
        self.v_prev_valid_mask = None

        # 跌倒偵測專用 Buffer
        self.f_pose_buf = deque(maxlen=seq_len)
        self.f_prev_raw_pose = None

        # 共用前後 5 秒影片緩衝與錄影狀態
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

    def _select_main_person(self, people):
        selected_person = None
        largest_area = 0.0

        for person in people:
            valid_points = [
                [person[kp_index][0], person[kp_index][1]]
                for kp_index in self.SELECTED_POINTS
                if person[kp_index][2] >= self.conf_thresh
            ]
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
            if valid[self.LEFT_HIP] and valid[self.RIGHT_HIP]:
                center = (points[self.LEFT_HIP] + points[self.RIGHT_HIP]) / 2.0
            elif valid[self.LEFT_SHOULDER] and valid[self.RIGHT_SHOULDER]:
                center = (points[self.LEFT_SHOULDER] + points[self.RIGHT_SHOULDER]) / 2.0
            else:
                center = np.mean(points[valid], axis=0)

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

    def process_frame(self, frame, wearable_status_text=""):
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

        # 5. 畫面疊加雙影像狀態 + 手環 36 規則表狀態
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

        if wearable_status_text:
            cv2.putText(
                annotated,
                f"Wearable (36-Rule): {wearable_status_text}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 220, 0),
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
# 3. 主執行流程 (背景執行緒跑手環 36 規則表 + 主執行緒跑單一 Camera 雙影像偵測)
# ----------------------------------------------------
if __name__ == "__main__":
    from pipeline_stream import WearableStreamWorker

    print("=== 1. 初始化多模態穿戴式預測器 (PPG + 隊友 EDA + 隊友 IMU) ===")
    wearable_predictor = MultimodalWearablePredictor(weights_dir="weights")
    detector = AnomalyDetector()

    # 啟動背景手環串流執行緒 (與 Camera 平行運作，互不卡頓)
    stream_worker = WearableStreamWorker(
        predictor=wearable_predictor,
        detector=detector,
        interval_sec=2.0,
    )
    stream_thread = threading.Thread(target=stream_worker.run, daemon=True)
    stream_thread.start()

    print("\n=== 2. 初始化雙任務影像偵測器 (暴力 + 跌倒) ===")
    vision_detector = DualVisionAnomalyDetector(
        pose_model_path="weights/yolo11n-pose.pt",
        violence_model_path="weights/lstm_v6_dual_branch.keras",
        fall_model_path="weights/fall_lstm_model.keras",
        save_dir="event_videos",
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Camera Open:", cap.isOpened())
    if not cap.isOpened():
        stream_worker.stop()
        raise RuntimeError("無法開啟 Webcam")

    print("\n============================================================")
    print("系統啟動中：單一攝影機同步進行 [暴力判斷 + 跌倒判斷 + 36規則表融合]")
    print("按 Q 離開")
    print("============================================================")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("讀不到攝影機畫面")
                break

            # 取得背景執行緒最新的手環狀態 (不阻塞 Camera FPS)
            latest_wearable = stream_worker.get_latest_state()

            # 1. 執行單幀雙任務影像推論
            annotated_frame, vision_res = vision_detector.process_frame(
                frame,
                wearable_status_text=latest_wearable.get("final_decision", "Init"),
            )

            # 2. 當有異常事件影片儲存完成時，結合最新手環狀態送入 AnomalyDetector 通報
            if vision_res["saved_video_path"] is not None:
                combined_input = {
                    "ppg_state": latest_wearable.get("ppg_state", "NORMAL"),
                    "eda_state": latest_wearable.get("eda_state", "Normal"),
                    "imu_state": latest_wearable.get("imu_state", "calm"),
                    "violence_status": vision_res["violence_status"],
                    "fall_status": vision_res["fall_status"],
                    "saved_video_path": vision_res["saved_video_path"],
                }
                report = detector.evaluate_status(combined_input)
                send_alert_notification(report)

            # 3. 顯示合併後的即時畫面
            cv2.imshow("Unified Multimodal Detection (Vision + Wearable)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream_worker.stop()
        pending_video = vision_detector.flush_pending_recording()
        if pending_video is not None:
            print(f"已補存中斷前影片：{pending_video}")

        cap.release()
        cv2.destroyAllWindows()
        print("=== 系統已安全關閉 ===")
