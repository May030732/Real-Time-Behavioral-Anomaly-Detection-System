import cv2
import os


# ==========================================
# 1. 基本設定
# ==========================================

VIDEO_FILE = "fall11.mp4"

OUTPUT_FOLDER = "fall11"


# ==========================================
# 2. 建立輸出資料夾
# ==========================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ==========================================
# 3. 開啟影片
# ==========================================

cap = cv2.VideoCapture(
    VIDEO_FILE
)


if not cap.isOpened():

    print(
        "影片開啟失敗：",
        VIDEO_FILE
    )

    exit()


# ==========================================
# 4. 取得影片資訊
# ==========================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print("==============================")
print("影片：", VIDEO_FILE)
print("FPS：", fps)
print("總 Frame：", total_frames)
print("==============================")


# ==========================================
# 5. 一幀一幀存成圖片
# ==========================================

frame_id = 0


while True:

    ret, frame = cap.read()


    if not ret:

        break


    # --------------------------------------
    # 圖片檔名
    # --------------------------------------

    file_name = (
        f"frame_{frame_id:05d}.jpg"
    )


    file_path = os.path.join(
        OUTPUT_FOLDER,
        file_name
    )


    # --------------------------------------
    # 儲存圖片
    # --------------------------------------

    cv2.imwrite(
        file_path,
        frame
    )


    print(
        f"{frame_id + 1}/{total_frames}"
        f" | {file_name}"
    )


    frame_id += 1


# ==========================================
# 6. 關閉影片
# ==========================================

cap.release()


print()
print("==============================")
print("影片轉圖片完成")
print("輸出資料夾：", OUTPUT_FOLDER)
print("圖片數量：", frame_id)
print("==============================")