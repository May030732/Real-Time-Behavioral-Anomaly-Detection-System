
from pathlib import Path
import os


# =========================================================
# 1. 系統資料夾路徑
# =========================================================

# 503_整合系統
BASE_DIR = Path(__file__).resolve().parent

# 500_系統整合
PROJECT_DIR = BASE_DIR.parent

# 501_暴力辨識
VIOLENCE_DIR = PROJECT_DIR / "501_暴力辨識"

# 502_跌倒辨識
FALL_DIR = PROJECT_DIR / "502_跌倒辨識"


# =========================================================
# 2. 模型路徑
# =========================================================

POSE_MODEL_FILE = "yolo11n-pose.pt"

VIOLENCE_MODEL_FILE = (
    VIOLENCE_DIR / "lstm_v6_dual_branch.keras"
)

FALL_MODEL_FILE = (
    FALL_DIR / "fall_lstm_model.keras"
)


# =========================================================
# 3. 攝影機設定
# =========================================================

CAMERA_INDEX = 0


# =========================================================
# 4. 模型辨識設定
# =========================================================

SEQUENCE_LENGTH = 30

KEYPOINT_CONF_THRESHOLD = 0.5

VIOLENCE_THRESHOLD = 0.5

FALL_THRESHOLD = 0.7


# =========================================================
# 5. YOLO 13 個人體關鍵點
# =========================================================

SELECTED_POINTS = [
    0,       # Nose
    5, 6,    # Shoulders
    7, 8,    # Elbows
    9, 10,   # Wrists
    11, 12,  # Hips
    13, 14,  # Knees
    15, 16   # Ankles
]


# =========================================================
# 6. 暴力辨識正規化設定
# =========================================================

LEFT_SHOULDER = 1
RIGHT_SHOULDER = 2

LEFT_HIP = 7
RIGHT_HIP = 8

MIN_BODY_SCALE = 5.0

CLIP_MIN = -20.0
CLIP_MAX = 20.0


# =========================================================
# 7. 異常事件影片設定
# =========================================================

VIDEO_SAVE_FOLDER = BASE_DIR / "abnormal_videos"

PRE_EVENT_SECONDS = 5.0

POST_EVENT_SECONDS = 5.0

RECORD_COOLDOWN_SECONDS = 10.0

VIDEO_SAVE_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 8. Discord 上傳設定
# =========================================================

DISCORD_UPLOAD_ENABLED = False

DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    ""
)

DISCORD_TIMEOUT = 60


# =========================================================
# 9. 檢查模型路徑
# =========================================================

def check_model_paths():

    print("=" * 60)
    print("System Configuration Check")
    print("=" * 60)

    models = {
        "Violence Model": VIOLENCE_MODEL_FILE,
        "Fall Model": FALL_MODEL_FILE
    }

    all_exists = True

    for name, path in models.items():

        print()
        print(f"{name}:")
        print(path)

        if path.is_file():
            print("Status: OK")

        else:
            print("Status: NOT FOUND")
            all_exists = False

    print()
    print("Video Save Folder:")
    print(VIDEO_SAVE_FOLDER)

    print()
    print("=" * 60)

    if all_exists:
        print("All LSTM model paths are correct.")

    else:
        print("Please check model file locations.")

    return all_exists


if __name__ == "__main__":

    check_model_paths()
# =====================================================
# Discord 上傳設定
# =====================================================

DISCORD_UPLOAD_ENABLED = True

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1555802237027221564/UGeo5apvhBmYSivo-9ErF2OiF3Sh3EEgZGXnuylzWV9q358tRoDYv4kGkK17Xw2BEdx1"

DISCORD_TIMEOUT = 60