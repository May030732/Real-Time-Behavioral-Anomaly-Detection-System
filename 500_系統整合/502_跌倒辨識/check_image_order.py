import os
import re
from datetime import datetime


# ==========================================
# 1. 圖片資料夾
# ==========================================

IMAGE_FOLDER = "fall9"


# ==========================================
# 2. 取得所有圖片
# ==========================================

image_files = [
    file
    for file in os.listdir(IMAGE_FOLDER)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]


# ==========================================
# 3. 從檔名取得時間
# ==========================================
#
# 例如：
#
# 2018-07-04T12_04_17.738369.jpg
#
# 會轉成 datetime：
#
# 2018-07-04 12:04:17.738369
#
# ==========================================

def get_timestamp(filename):

    # 去掉副檔名
    name = os.path.splitext(filename)[0]

    # 尋找日期時間
    match = re.search(
        r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\.\d+)",
        name
    )

    if match is None:
        return datetime.max

    timestamp_text = match.group(1)

    try:

        timestamp = datetime.strptime(
            timestamp_text,
            "%Y-%m-%dT%H_%M_%S.%f"
        )

        return timestamp

    except ValueError:

        print(
            "⚠ 無法解析時間：",
            filename
        )

        return datetime.max


# ==========================================
# 4. 按真正時間排序
# ==========================================

image_files.sort(
    key=get_timestamp
)


# ==========================================
# 5. 顯示排序結果
# ==========================================

print()
print("==========================================")
print("Fall Dataset 圖片順序")
print("==========================================")

for index, filename in enumerate(image_files):

    timestamp = get_timestamp(
        filename
    )

    print(
        f"{index:03d}"
        f" | {timestamp.strftime('%H:%M:%S.%f')}"
        f" | {filename}"
    )


print("==========================================")
print(
    "總圖片數：",
    len(image_files)
)
print("==========================================")