import time

# 支援模組匯入與直接執行
if __package__:
    from .rule_table import get_final_decision
else:
    from rule_table import get_final_decision


class AnomalyDetector:
    """
    PPG、EDA、IMU 共 36 種狀態組合的異常監控評估器。
    """

    def evaluate_status(self, inference_result):
        """
        根據感測器狀態重新查詢 36 種規則表，
        產生手環判斷結果及告警等級。
        """
        ppg_state = inference_result.get("ppg_state", "Unknown")
        eda_state = inference_result.get("eda_state", "Unknown")
        imu_state = inference_result.get("imu_state", "Unknown")

        final_decision = get_final_decision(
            ppg_state,
            eda_state,
            imu_state,
        )

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        reasons = [
            f"三模態狀態組合: "
            f"[PPG: {ppg_state}] + "
            f"[EDA: {eda_state}] + "
            f"[IMU: {imu_state}]"
        ]

        if final_decision == "Abnormal":
            status = "ANOMALY"
            alert_level = "CRITICAL"
            reasons.append(
                "符合 36 種組合表【Abnormal】規則，"
                "手環資料判定為異常。"
            )

        elif final_decision == "Suspected_Abnormal":
            status = "WARNING"
            alert_level = "MEDIUM"
            reasons.append(
                "符合 36 種組合表【Suspected_Abnormal】規則，"
                "進入疑似異常觀察階段。"
            )

        elif final_decision == "Normal":
            status = "NORMAL"
            alert_level = "INFO"
            reasons.append(
                "符合 36 種組合表【Normal】規則，"
                "手環資料判定為正常。"
            )

        else:
            status = "UNKNOWN"
            alert_level = "LOW"
            reasons.append(
                "感測器狀態缺失或名稱未定義，無法判斷。"
            )

        return {
            "timestamp": timestamp,
            "status": status,
            "alert_level": alert_level,
            "final_decision": final_decision,
            "ppg_state": ppg_state,
            "eda_state": eda_state,
            "imu_state": imu_state,
            "reasons": reasons,
        }

    def evaluate_cross_validation(self, inference_result):
        """保留原本的呼叫介面。"""
        return self.evaluate_status(inference_result)


def send_alert_notification(report):
    """
    將評估結果輸出至主控台。
    目前尚未包含 Discord 傳送功能。
    """
    status = report["status"]
    timestamp = report["timestamp"]
    final_decision = report["final_decision"]

    if status == "ANOMALY":
        print("\n🚨【手環警告：36 種組合表判定異常】")
        print(
            f"⏰ 時間: {timestamp} | "
            f"警告等級: {report['alert_level']}"
        )
        print(
            f"📊 狀態組合: "
            f"PPG={report['ppg_state']} | "
            f"EDA={report['eda_state']} | "
            f"IMU={report['imu_state']}"
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
            f"✓ [{timestamp}] 手環資料正常 ({final_decision}) | "
            f"{report['reasons'][0]}"
        )

    else:
        print(
            f"❓ [{timestamp}] 無法判斷 ({final_decision}) | "
            f"{report['reasons'][0]}"
        )
        for reason in report["reasons"][1:]:
            print(f"   - {reason}")


if __name__ == "__main__":
    detector = AnomalyDetector()

    # 測試資料不必指定 final_decision，由規則表決定
    test_cases = [
        (
            "正常",
            {
                "ppg_state": "NORMAL",
                "eda_state": "Normal",
                "imu_state": "calm",
            },
            "NORMAL",
        ),
        (
            "疑似異常",
            {
                "ppg_state": "PERSISTENT_HIGH_HR",
                "eda_state": "Normal",
                "imu_state": "calm",
            },
            "WARNING",
        ),
        (
            "異常",
            {
                "ppg_state": "CRITICAL_EXTREME_HR",
                "eda_state": "High_Arousal",
                "imu_state": "fierce",
            },
            "ANOMALY",
        ),
        (
            "資料缺失",
            {
                "eda_state": "Normal",
                "imu_state": "calm",
            },
            "UNKNOWN",
        ),
    ]

    for name, data, expected_status in test_cases:
        print(f"\n=== 測試：{name} ===")
        report = detector.evaluate_status(data)
        assert report["status"] == expected_status, report
        send_alert_notification(report)

    print("\n✅ 四種報告狀態測試通過")