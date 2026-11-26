
import cv2
import time

from collections import deque
from datetime import datetime

from config import (
    VIDEO_SAVE_FOLDER,
    PRE_EVENT_SECONDS,
    POST_EVENT_SECONDS,
    RECORD_COOLDOWN_SECONDS
)


class EventRecorder:

    # =====================================================
    # 1. 初始化
    # =====================================================

    def __init__(self):

        self.video_buffer = deque()

        self.recording = False

        self.event_frames = []
        self.event_times = []

        self.event_types = set()

        self.recording_end_time = 0.0

        self.last_saved_time = float("-inf")

        self.previous_abnormal = False

        VIDEO_SAVE_FOLDER.mkdir(
            parents=True,
            exist_ok=True
        )

        print("Event Recorder Ready.")


    # =====================================================
    # 2. 更新前五秒 Buffer
    # =====================================================

    def update_buffer(self, frame, current_time):

        self.video_buffer.append(
            (current_time, frame)
        )

        while self.video_buffer:

            oldest_time = self.video_buffer[0][0]

            if (
                current_time - oldest_time
                > PRE_EVENT_SECONDS
            ):

                self.video_buffer.popleft()

            else:

                break


    # =====================================================
    # 3. 開始異常事件錄影
    # =====================================================

    def start_event(
        self,
        current_time,
        event_types
    ):

        self.recording = True

        self.event_frames = [
            frame
            for _, frame in self.video_buffer
        ]

        self.event_times = [
            timestamp
            for timestamp, _ in self.video_buffer
        ]

        self.event_types = set(event_types)

        self.recording_end_time = (
            current_time + POST_EVENT_SECONDS
        )

        print()
        print("=" * 60)
        print("Abnormal Event Detected")
        print("Event:", ", ".join(sorted(self.event_types)))
        print("Recording next 5 seconds...")
        print("=" * 60)


    # =====================================================
    # 4. 儲存 MP4
    # =====================================================

    def save_video(self):

        if len(self.event_frames) < 2:

            print("Not enough frames.")

            return None

        duration = (
            self.event_times[-1]
            - self.event_times[0]
        )

        if duration <= 0:

            return None

        fps = (
            len(self.event_frames) - 1
        ) / duration

        fps = max(1.0, min(fps, 60.0))

        height, width = (
            self.event_frames[0].shape[:2]
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        event_name = "_".join(
            sorted(self.event_types)
        ).lower()

        filename = (
            f"{event_name}_{timestamp}.mp4"
        )

        output_path = (
            VIDEO_SAVE_FOLDER / filename
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():

            writer.release()

            print("Video Writer Open Failed.")

            return None

        try:

            for frame in self.event_frames:

                writer.write(frame)

        finally:

            writer.release()

        if (
            not output_path.is_file()
            or output_path.stat().st_size == 0
        ):

            print("Video Save Failed.")

            return None

        print()
        print("=" * 60)
        print("Abnormal Event Video Saved")
        print("=" * 60)
        print("Event:", event_name)
        print("Video:", output_path)
        print(f"Duration: {duration:.2f} seconds")
        print(f"FPS: {fps:.2f}")
        print("Frames:", len(self.event_frames))
        print("=" * 60)

        return str(output_path)


    # =====================================================
    # 5. 每幀更新
    # =====================================================

    def update(
        self,
        frame,
        violence_status,
        fall_status
    ):

        current_time = time.monotonic()

        frame_copy = frame.copy()

        event_types = set()

        if violence_status == "VIOLENCE":

            event_types.add("VIOLENCE")

        if fall_status == "FALL":

            event_types.add("FALL")

        abnormal = bool(event_types)

        # 持續維護前五秒影像
        self.update_buffer(
            frame_copy,
            current_time
        )

        saved_path = None

        # -------------------------------------------------
        # 正在錄製異常事件
        # -------------------------------------------------

        if self.recording:

            self.event_frames.append(
                frame_copy
            )

            self.event_times.append(
                current_time
            )

            self.event_types.update(
                event_types
            )

            if (
                current_time
                >= self.recording_end_time
            ):

                saved_path = self.save_video()

                self.recording = False

                self.last_saved_time = (
                    time.monotonic()
                )

                self.event_frames.clear()
                self.event_times.clear()
                self.event_types.clear()

        # -------------------------------------------------
        # 尚未錄製異常事件
        # -------------------------------------------------

        else:

            cooldown_finished = (
                current_time - self.last_saved_time
                >= RECORD_COOLDOWN_SECONDS
            )

            new_event = (
                abnormal
                and not self.previous_abnormal
            )

            if new_event and cooldown_finished:

                self.start_event(
                    current_time,
                    event_types
                )

        self.previous_abnormal = abnormal

        return saved_path


    # =====================================================
    # 6. 取得錄影狀態
    # =====================================================

    def get_status(self):

        if self.recording:

            remaining = max(
                0.0,
                self.recording_end_time
                - time.monotonic()
            )

            return f"RECORDING: {remaining:.1f}s"

        cooldown_remaining = (
            RECORD_COOLDOWN_SECONDS
            - (
                time.monotonic()
                - self.last_saved_time
            )
        )

        if cooldown_remaining > 0:

            return (
                f"COOLDOWN: "
                f"{cooldown_remaining:.1f}s"
            )

        return "STANDBY"


    # =====================================================
    # 7. 關閉程式時儲存未完成的影片
    # =====================================================

    def close(self):

        saved_path = None

        if self.recording:

            print("Saving incomplete event video...")

            saved_path = self.save_video()

        self.recording = False

        self.event_frames.clear()
        self.event_times.clear()
        self.event_types.clear()
        self.video_buffer.clear()

        return saved_path


# =========================================================
# 8. 獨立初始化測試
# =========================================================

if __name__ == "__main__":

    recorder = EventRecorder()

    print("Video Folder:", VIDEO_SAVE_FOLDER)

    print("Status:", recorder.get_status())
