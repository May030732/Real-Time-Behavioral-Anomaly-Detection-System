import numpy as np

# 改成你要查看的檔案路徑
file_path = r"C:\fall_detection_project\fall_detection\datasets\fall11_dataset.npz"

with np.load(file_path) as data:
    print("檔案裡有哪些陣列：", data.files)

    X = data["X"]
    y = data["y"]

    print("\nX 的形狀：", X.shape)
    print("y 的形狀：", y.shape)

    print("\n跌倒樣本數：", np.sum(y == 1))
    print("非跌倒樣本數：", np.sum(y == 0))

    if len(X) > 0:
        print("\n第一筆樣本的形狀：", X[0].shape)
        print("第一筆樣本的標籤：", y[0])
        print("第一筆樣本、第一幀的特徵：")
        print(X[0, 0])