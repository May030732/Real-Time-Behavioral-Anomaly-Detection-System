from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-pose.pt")

# 先試內建相機
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera Open:", cap.isOpened())

while True:
    ret, frame = cap.read()

    if not ret:
        print("讀不到影像")
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("YOLO11 Pose", annotated)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()