from ultralytics import YOLO
import cv2
import csv

# 載入 YOLO11 Pose 模型
model = YOLO("yolo11n-pose.pt")

# 開啟電腦攝影機
# CAP_DSHOW：Windows 使用 DirectShow，可加快開啟速度
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 確認攝影機是否成功開啟
print("Camera Open:", cap.isOpened())

# 要保留的13個骨架點
selected_points = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# ==========================
# CSV 欄位名稱
# 每一列：
# Frame + 13個點(x,y)
# 共27欄
# ==========================
header = [
    "frame",
    "nose_x", "nose_y",
    "left_shoulder_x", "left_shoulder_y",
    "right_shoulder_x", "right_shoulder_y",
    "left_elbow_x", "left_elbow_y",
    "right_elbow_x", "right_elbow_y",
    "left_wrist_x", "left_wrist_y",
    "right_wrist_x", "right_wrist_y",
    "left_hip_x", "left_hip_y",
    "right_hip_x", "right_hip_y",
    "left_knee_x", "left_knee_y",
    "right_knee_x", "right_knee_y",
    "left_ankle_x", "left_ankle_y",
    "right_ankle_x", "right_ankle_y"
]

# ==========================
# 建立 CSV 檔案
# newline=""：避免空白列
# utf-8-sig：Excel 可正常開啟中文
# ==========================
csv_file = open("pose_data.csv", "w", newline="", encoding="utf-8-sig")

writer = csv.writer(csv_file)

# 寫入第一列(欄位名稱)
writer.writerow(header)

# 目前影格編號
frame_id = 0

# ==========================
# 持續讀取攝影機畫面
# ==========================
while True:

    # 讀取一張影像
    ret, frame = cap.read()

    # 如果讀不到畫面就離開
    if not ret:
        print("讀不到影像")
        break

    # 使用 YOLO11 Pose 偵測人體姿態
    results = model(frame, verbose=False)

    # ==========================
    # 確認有偵測到人體骨架
    # ==========================
    if len(results) > 0 and results[0].keypoints is not None:

        # 取得所有人物17個關鍵點座標
        # shape = (人物數,17,2)
        xy = results[0].keypoints.xy.cpu().numpy()

        # 取得17個關鍵點信心值
        # shape = (人物數,17)
        conf = results[0].keypoints.conf.cpu().numpy()

        # 目前只取第一位人物
        if len(xy) > 0:

            # 建立一列資料
            # 第一欄放 Frame 編號
            row = [frame_id]

            # 依序擷取13個關鍵點
            for idx in selected_points:

                # 若信心值低於0.5，表示偵測不可靠
                if conf[0][idx] < 0.5:

                    row.extend([-1, -1])  
    #空白值 "" 用 pandas 讀取時會變成 NaN。
    #有些深度學習框架不能直接處理 NaN。
    #使用固定值（例如 -1）會比較容易做後續資料前處理。
                else:

                    # 取得 x、y 座標
                    x = round(float(xy[0][idx][0]), 2)
                    y = round(float(xy[0][idx][1]), 2)

                    # 加入CSV資料
                    row.extend([x, y])

            # 將一整列寫入CSV
            writer.writerow(row)

    # ==========================
    # 顯示YOLO偵測畫面
    # （畫面仍為17點，不影響CSV）
    # ==========================
    annotated = results[0].plot()

    cv2.imshow("YOLO11 Pose", annotated)

    # Frame編號加1
    frame_id += 1

    # 按 q 離開
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# 關閉攝影機
# ==========================
cap.release()

# 關閉CSV檔案
csv_file.close()

# 關閉所有OpenCV視窗
cv2.destroyAllWindows()

print("CSV 已成功儲存：pose_data.csv")