import os
import pandas as pd

# ==========================================
# 1. 設定要檢查的 CSV 檔案路徑
# ==========================================
# 請將這裡替換成你想查看的 CSV 檔案路徑
csv_path = os.path.join("Heart_Rate_Stress", "Train Data", "Train Data Zip", "time_domain_features_train.csv")

def inspect_csv(file_path):
    print("=" * 60)
    print(f"🔍 正在檢查檔案：{file_path}")
    print("=" * 60)

    # 檢查檔案是否存在
    if not os.path.exists(file_path):
        print(f"❌ 錯誤：找不到檔案【{file_path}】！請檢查路徑名稱是否正確。")
        return

    try:
        # 讀取 CSV
        df = pd.read_csv(file_path)
        
        # 1. 基本維度資訊
        rows, cols = df.shape
        print(f"📊 資料規模：共 {rows:,} 筆資料（列），{cols} 個欄位（行）\n")

        # 2. 列出所有欄位名稱與資料型態
        print("📋 欄位清單 (Column Names & Types):")
        for idx, (col_name, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
            # 檢查是否有缺失值 (NaN)
            null_count = df[col_name].isnull().sum()
            null_info = f"(⚠️ 含 {null_count} 個缺失值)" if null_count > 0 else ""
            print(f"  [{idx:02d}] {col_name:<25} | 型態: {str(dtype):<10} {null_info}")

        # 3. 印出前 5 筆資料預覽
        print("\n" + "-" * 60)
        print("👀 前 5 筆資料預覽 (Head 5):")
        print("-" * 60)
        pd.set_option('display.max_columns', None)  # 讓所有欄位完整顯示不折疊
        pd.set_option('display.width', 1000)
        print(df.head())

        # 4. 如果有 condition 欄位，順便印出類別分布
        if 'condition' in df.columns:
            print("\n" + "-" * 60)
            print("🎯 目標標籤 (condition) 分布：")
            print(df['condition'].value_counts())

        print("\n" + "=" * 60)
        print("✅ 檢查完成！")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 讀取 CSV 時發生錯誤：{e}")

if __name__ == "__main__":
    inspect_csv(csv_path)