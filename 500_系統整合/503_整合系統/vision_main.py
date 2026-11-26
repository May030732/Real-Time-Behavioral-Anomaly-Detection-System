
import cv2
import numpy as np
import time

from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

from violence_detector import ViolenceDetector
from fall_detector import FallDetector
from event_recorder import EventRecorder
from discord_uploader import DiscordUploader


# =====================================================
# 1. 基本設定
# =====================================================

POSE_MODEL_FILE = "yolo11n-pose.pt"

CAMERA_INDEX = 0

WINDOW_NAME = "Integrated Detection System"


# =====================================================
# 2. 擷取 YOLO 人體骨架
# =====================================================

def extract_people(results):

    if len(results) == 0:
        return []

    keypoints = results[0].keypoints

    if keypoints is None:
        return []

    if keypoints.conf is None:
        return []

    xy = keypoints.xy.cpu().numpy()

    conf = keypoints.conf.cpu().numpy()

    if len(xy) == 0:
        return []

    people = np.concatenate(
        [
            xy,
            conf[..., np.newaxis]
        ],
        axis=2
    )

    return people.astype(np.float32)


# =====================================================
# 3. 顯示辨識結果
# =====================================================

def draw_status(
    frame,
    label,
    status,
    probability,
    position
):

    if status in ("VIOLENCE", "FALL"):

        color = (0, 0, 255)

    elif status in (
        "NON_VIOLENCE",
        "NON_FALL"
    ):

        color = (0, 255, 0)

    else:

        color = (0, 255, 255)

    if probability is None:

        message = f"{label}: {status}"

    else:

        message = (
            f"{label}: {status} "
            f"({probability:.3f})"
        )

    cv2.putText(
        frame,
        message,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )


# =====================================================
# 4. 系統綜合判斷
# =====================================================

def get_system_status(
    violence_status,
    fall_status,
    people
):

    if (
        violence_status == "VIOLENCE"
        and fall_status == "FALL"
    ):

        return "VIOLENCE + FALL", (0, 0, 255)

    elif violence_status == "VIOLENCE":

        return "VIOLENCE", (0, 0, 255)

    elif fall_status == "FALL":

        return "FALL", (0, 0, 255)

    elif (
        violence_status == "NON_VIOLENCE"
        and fall_status == "NON_FALL"
    ):

        return "NORMAL", (0, 255, 0)

    elif len(people) == 0:

        return "NO_PERSON", (0, 255, 255)

    else:

        return "COLLECTING", (0, 255, 255)


# =====================================================
# 5. Discord 背景上傳
# =====================================================

def upload_video(uploader, video_path):

    try:

        success = uploader.upload(video_path)

        if success:

            print("Event Upload Complete.")

        else:

            print("Event Upload Failed.")
            print("Local Video Preserved.")

    except Exception as error:

        print(
            "Upload Error:",
            type(error).__name__
        )

        print("Local Video Preserved.")


# =====================================================
# 6. 主程式
# =====================================================

def main():

    print("=" * 60)
    print("Integrated Abnormal Behavior Detection")
    print("=" * 60)

    # -------------------------------------------------
    # 載入 YOLO Pose
    # -------------------------------------------------

    print("Loading YOLO Pose...")

    pose_model = YOLO(
        POSE_MODEL_FILE
    )

    # -------------------------------------------------
    # 載入暴力辨識模型
    # -------------------------------------------------

    violence_detector = ViolenceDetector()

    # -------------------------------------------------
    # 載入跌倒辨識模型
    # -------------------------------------------------

    fall_detector = FallDetector()

    # -------------------------------------------------
    # 初始化異常錄影模組
    # -------------------------------------------------

    event_recorder = EventRecorder()

    # -------------------------------------------------
    # 初始化 Discord 上傳模組
    # -------------------------------------------------

    discord_uploader = DiscordUploader()

    # -------------------------------------------------
    # 開啟攝影機
    # -------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print("Camera Open Failed.")

        cap.release()

        return

    # -------------------------------------------------
    # 建立背景上傳執行緒
    # -------------------------------------------------

    upload_executor = ThreadPoolExecutor(
        max_workers=1
    )

    upload_tasks = []

    print()
    print("Camera Open Successfully.")
    print("Integrated System Ready.")
    print("Press Q or ESC to Exit.")

    previous_time = time.monotonic()

    try:

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL
        )

        # =============================================
        # 7. 即時辨識迴圈
        # =============================================

        while True:

            ret, frame = cap.read()

            if not ret:

                print("Cannot Read Camera.")

                break

            # -----------------------------------------
            # YOLO Pose：每幀只執行一次
            # -----------------------------------------

            results = pose_model(
                frame,
                verbose=False
            )

            annotated = results[0].plot()

            people = extract_people(
                results
            )

            # -----------------------------------------
            # 暴力辨識
            # -----------------------------------------

            violence_status, violence_probability = (
                violence_detector.update(
                    people
                )
            )

            # -----------------------------------------
            # 跌倒辨識
            # -----------------------------------------

            fall_status, fall_probability = (
                fall_detector.update(
                    people
                )
            )

            # -----------------------------------------
            # 系統綜合判斷
            # -----------------------------------------

            system_status, system_color = (
                get_system_status(
                    violence_status,
                    fall_status,
                    people
                )
            )

            # -----------------------------------------
            # 顯示暴力辨識結果
            # -----------------------------------------

            draw_status(
                annotated,
                "Violence",
                violence_status,
                violence_probability,
                (20, 35)
            )

            # -----------------------------------------
            # 顯示跌倒辨識結果
            # -----------------------------------------

            draw_status(
                annotated,
                "Fall",
                fall_status,
                fall_probability,
                (20, 70)
            )

            # -----------------------------------------
            # 顯示系統綜合結果
            # -----------------------------------------

            cv2.putText(
                annotated,
                f"System: {system_status}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                system_color,
                2
            )

            # -----------------------------------------
            # 計算 FPS
            # -----------------------------------------

            current_time = time.monotonic()

            fps = 1.0 / max(
                current_time - previous_time,
                0.000001
            )

            previous_time = current_time

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            # =========================================
            # 8. 異常事件錄影
            # =========================================

            saved_path = event_recorder.update(
                annotated,
                violence_status,
                fall_status
            )

            # =========================================
            # 9. 影片完成後背景上傳 Discord
            # =========================================

            if saved_path is not None:

                task = upload_executor.submit(
                    upload_video,
                    discord_uploader,
                    saved_path
                )

                upload_tasks.append(task)

            # 清除已完成的上傳工作

            upload_tasks = [
                task
                for task in upload_tasks
                if not task.done()
            ]

            # =========================================
            # 10. 顯示錄影狀態
            # =========================================

            recorder_status = (
                event_recorder.get_status()
            )

            if recorder_status.startswith(
                "RECORDING"
            ):

                recorder_color = (0, 0, 255)

            elif recorder_status.startswith(
                "COOLDOWN"
            ):

                recorder_color = (0, 165, 255)

            else:

                recorder_color = (0, 255, 0)

            cv2.putText(
                annotated,
                f"Recorder: {recorder_status}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                recorder_color,
                2
            )

            # =========================================
            # 11. 顯示 Discord 上傳狀態
            # =========================================

            if len(upload_tasks) > 0:

                upload_status = (
                    f"UPLOADING / QUEUED: "
                    f"{len(upload_tasks)}"
                )

                upload_color = (255, 255, 0)

            else:

                upload_status = "IDLE"

                upload_color = (0, 255, 0)

            cv2.putText(
                annotated,
                f"Discord: {upload_status}",
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                upload_color,
                2
            )

            # =========================================
            # 12. 顯示攝影機畫面
            # =========================================

            cv2.imshow(
                WINDOW_NAME,
                annotated
            )

            # =========================================
            # 13. 結束程式
            # =========================================

            key = cv2.waitKey(1) & 0xFF

            # 支援 q、Q、ESC

            if key in (
                ord("q"),
                ord("Q"),
                27
            ):

                break

            # 支援視窗右上角 X

            try:

                if cv2.getWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_VISIBLE
                ) < 1:

                    break

            except cv2.error:

                break

    except KeyboardInterrupt:

        print("Keyboard Interrupt.")

    finally:

        # =============================================
        # 14. 關閉攝影機
        # =============================================

        cap.release()

        cv2.destroyAllWindows()

        try:

            # 儲存尚未完成的異常影片

            remaining_video = (
                event_recorder.close()
            )

            if remaining_video is not None:

                upload_executor.submit(
                    upload_video,
                    discord_uploader,
                    remaining_video
                )

        finally:

            # 等待剩餘 Discord 上傳工作完成

            print("Waiting for pending uploads...")

            upload_executor.shutdown(
                wait=True
            )

            print("System Closed.")


# =====================================================
# 15. 執行主程式
# =====================================================

if __name__ == "__main__":

    main()
