import time

# 支援模組匯入與直接執行
if __package__:
    from .rule_table import get_final_decision
else:
    from rule_table import get_final_decision


class AnomalyDetector:
    """
    整合【PPG/EDA/IMU 36 種狀態組合】與【影像暴力/跌倒偵測】的異常監控評估器。
    """

    def evaluate_status(self, inference_result):
        """
        根據感測器狀態查詢 36 種規則表，並結合影像(暴力/跌倒)判定產生最終告警等級。
        """
        ppg_state = inference_result.get("ppg_state", "Unknown")
        eda_state = inference_result.get("eda_state", "Unknown")
        imu_state = inference_result.get("imu_state", "Unknown")

        # 影像狀態 (若無傳入則預設為安全狀態，相容純手環測試)
        violence_status = inference_result.get("violence_status", "NON_VIOLENCE")
        fall_status = inference_result.get("fall_status", "NON_FALL")
        saved_video_path = inference_result.get("saved_video_path", None)

        # 若 EDA 還在收集基線或無效值，查 36 表時暫以 "Normal" 代入
        eda_for_table = (
            eda_state
            if eda_state in {"Normal", "Suspected_Arousal", "High_Arousal"}
            else "Normal"
        )

        wearable_decision = get_final_decision(
            ppg_state,
            eda_for_table,
            imu_state,
        )

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        reasons = [
            f"三模態狀態組合: "
            f"[PPG: {ppg_state}] + "
            f"[EDA: {eda_state}] + "
            f"[IMU: {imu_state}] -> 手環判定: {wearable_decision}"
        ]

        vision_anomaly = (violence_status == "VIOLENCE") or (fall_status == "FALL")

        # 1. 最高優先級：影像明確偵測到暴力或跌倒，或 36 規則表判定為 Abnormal
        if vision_anomaly or wearable_decision == "Abnormal":
            status = "ANOMALY"
            alert_level = "CRITICAL"
            final_decision = "Abnormal"

            if violence_status == "VIOLENCE":
                reasons.append("【影像異常】偵測到暴力行為 (VIOLENCE)！")
            if fall_status == "FALL":
                reasons.append("【影像異常】偵測到跌倒事件 (FALL)！")
            if wearable_decision == "Abnormal":
                reasons.append("符合 36 種組合表【Abnormal】規則，手環資料判定為異常。")
            if saved_video_path:
                reasons.append(f"異常事件影片已儲存：{saved_video_path}")

        # 2. 次高優先級：36 規則表判定為 Suspected_Abnormal
        elif wearable_decision == "Suspected_Abnormal":
            status = "WARNING"
            alert_level = "MEDIUM"
            final_decision = "Suspected_Abnormal"
            reasons.append(
                "符合 36 種組合表【Suspected_Abnormal】規則，進入疑似異常觀察階段。"
            )

        # 3. 正常狀態
        elif wearable_decision == "Normal":
            status = "NORMAL"
            alert_level = "INFO"
            final_decision = "Normal"
            reasons.append(
                "符合 36 種組合表【Normal】規則，手環與影像狀態皆正常。"
            )

        else:
            status = "UNKNOWN"
            alert_level = "LOW"
            final_decision = "Unknown"
            reasons.append("感測器狀態缺失或名稱未定義，無法判斷。")

        return {
            "timestamp": timestamp,
            "status": status,
            "alert_level": alert_level,
            "final_decision": final_decision,
            "wearable_decision": wearable_decision,
            "ppg_state": ppg_state,
            "eda_state": eda_state,
            "imu_state": imu_state,
            "violence_status": violence_status,
            "fall_status": fall_status,
            "saved_video_path": saved_video_path,
            "reasons": reasons,
        }

    def evaluate_cross_validation(self, inference_result):
        """保留原本的呼叫介面。"""
        return self.evaluate_status(inference_result)


def send_alert_notification(report):
    """
    將評估結果輸出至主控台 (可在此串接 discord_notifier.py 上傳影片)。
    """
    status = report["status"]
    timestamp = report["timestamp"]
    final_decision = report["final_decision"]

    if status == "ANOMALY":
        print("\n🚨【系統警告：偵測到異常事件】")
        print(f"⏰ 時間: {timestamp} | 警告等級: {report['alert_level']}")
        print(
            f"📊 手環狀態: PPG={report['ppg_state']} | "
            f"EDA={report['eda_state']} | IMU={report['imu_state']}"
        )
        print(
            f"📷 影像狀態: Violence={report.get('violence_status', 'N/A')} | "
            f"Fall={report.get('fall_status', 'N/A')}"
        )
        print("判定細節:")
        for reason in report["reasons"]:
            print(f"   - {reason}")
        print("--------------------------------------------------\n")

    elif status == "WARNING":
        print(
            f"⚠️ [{timestamp}] 疑似異常 ({final_decision}) | "
            f"{report['reasons'][0]}"
        )

    elif status == "NORMAL":
        print(
            f"✓ [{timestamp}] 系統狀態正常 ({final_decision}) | "
            f"{report['reasons'][0]}"
        )

    else:
        print(
            f"❓ [{timestamp}] 無法判斷 ({final_decision}) | "
            f"{report['reasons'][0]}"
        )
        for reason in report["reasons"][1:]:
            print(f"   - {reason}")
