import time

class AnomalyDetector:
    """
    三模態 (PPG, EDA, IMU) 27 種真值表查表與異常監控評估器
    """
    def __init__(self):
        pass

    def evaluate_status(self, inference_result):
        """
        根據 27 種真值表查表出的 final_decision 進行告警層級評估
        :param inference_result: 來自 main_inference.py 輸出的字典結果
        """
        ppg_state = inference_result.get("ppg_state", "Unknown")
        eda_state = inference_result.get("eda_state", "Unknown")
        imu_state = inference_result.get("imu_state", "Unknown")
        final_decision = inference_result.get("final_decision", "Unknown")

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        reasons = [f"三模態狀態組合: [PPG: {ppg_state}] + [EDA: {eda_state}] + [IMU: {imu_state}]"]

        # 根據 27 種真值表的最終判定，給予對應的告警狀態與等級
        if final_decision == "Abnormal":
            status = "ANOMALY"
            alert_level = "CRITICAL"
            reasons.append("觸發 27 種真值表【Abnormal】規則，系統確認數據異常！")
        elif final_decision == "Suspected_Abnormal":
            status = "WARNING"
            alert_level = "MEDIUM"
            reasons.append("觸發 27 種真值表【Suspected_Abnormal】規則，進入疑似異常觀察階段。")
        elif final_decision == "Normal":
            status = "NORMAL"
            alert_level = "INFO"
            reasons.append("符合 27 種真值表【Normal】規則，狀態穩定。")
        else:
            status = "UNKNOWN"
            alert_level = "LOW"
            reasons.append(f"⚠️ 未知的狀態判定結果: {final_decision}")

        return {
            "timestamp": timestamp,
            "status": status,             # NORMAL, WARNING, ANOMALY
            "alert_level": alert_level,   # INFO, MEDIUM, CRITICAL
            "final_decision": final_decision,
            "ppg_state": ppg_state,
            "eda_state": eda_state,
            "imu_state": imu_state,
            "reasons": reasons
        }

    def evaluate_cross_validation(self, inference_result):
        """相容新版命名介面之轉接函式"""
        return self.evaluate_status(inference_result)


def send_alert_notification(report):
    """
    異常通報與主控台日誌輸出函式
    """
    status = report["status"]
    timestamp = report["timestamp"]
    final_decision = report["final_decision"]
    
    if status == "ANOMALY":
        print("\n🚨🚨 【系統警告：27種真值表確認數據異常 (Abnormal)】 🚨🚨")
        print(f"⏰ 時間: {timestamp} | 警告等級: {report['alert_level']}")
        print(f"📊 狀態組合: PPG={report['ppg_state']} | EDA={report['eda_state']} | IMU={report['imu_state']}")
        print("⚠️ 判定細節:")
        for r in report["reasons"]:
            print(f"   - {r}")
        print("--------------------------------------------------\n")
        
    elif status == "WARNING":
        print(f"⚠️ [{timestamp}] 疑似異常提示 ({final_decision}): {report['reasons'][0]}")
        
    else:
        print(f"✓ [{timestamp}] 數據正常 ({final_decision}) | {report['reasons'][0]}")


# 模組測試用程式區塊
if __name__ == "__main__":
    detector = AnomalyDetector()
    
    # 測試 A：正常情況
    mock_res_normal = {
        "ppg_state": "Medium",
        "eda_state": "Normal",
        "imu_state": "calm",
        "final_decision": "Normal"
    }
    rep_normal = detector.evaluate_status(mock_res_normal)
    send_alert_notification(rep_normal)

    # 測試 B：嚴重異常情況 (如第 27 組 Fast + High_Arousal + fierce)
    mock_res_anomaly = {
        "ppg_state": "Fast",
        "eda_state": "High_Arousal",
        "imu_state": "fierce",
        "final_decision": "Abnormal"
    }
    rep_anomaly = detector.evaluate_status(mock_res_anomaly)
    send_alert_notification(rep_anomaly)