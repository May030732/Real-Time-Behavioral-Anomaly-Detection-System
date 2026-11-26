
import requests

from pathlib import Path
from datetime import datetime

from config import (
    DISCORD_UPLOAD_ENABLED,
    DISCORD_WEBHOOK_URL,
    DISCORD_TIMEOUT
)


class DiscordUploader:

    # =====================================================
    # 1. 初始化
    # =====================================================

    def __init__(self):

        self.enabled = DISCORD_UPLOAD_ENABLED

        self.webhook_url = DISCORD_WEBHOOK_URL

        self.timeout = DISCORD_TIMEOUT

        print("Discord Uploader Ready.")

        if self.enabled:
            print("Discord Upload: ENABLED")
        else:
            print("Discord Upload: DISABLED")


    # =====================================================
    # 2. 判斷異常事件類型
    # =====================================================

    def get_event_type(self, video_path):

        filename = Path(video_path).name.lower()

        if "fall" in filename and "violence" in filename:

            return "暴力與跌倒事件"

        elif "violence" in filename:

            return "暴力事件"

        elif "fall" in filename:

            return "跌倒事件"

        else:

            return "異常事件"


    # =====================================================
    # 3. 上傳異常影片
    # =====================================================

    def upload(self, video_path):

        if not self.enabled:

            print("Discord Upload Disabled.")

            return False

        if not self.webhook_url.startswith(
            "https://discord.com/api/webhooks/"
        ):

            print("Invalid Discord Webhook URL.")

            return False

        video_path = Path(video_path)

        if not video_path.is_file():

            print("Video File Not Found.")

            return False

        event_type = self.get_event_type(video_path)

        current_time = datetime.now().strftime(
            "%Y/%m/%d %H:%M:%S"
        )

        message = (
            "⚠️ 長照機構異常行為警報\n"
            f"事件類型：{event_type}\n"
            f"通報時間：{current_time}\n"
            "請相關人員確認事件影片。"
        )

        print()
        print("=" * 60)
        print("Uploading Video to Discord...")
        print("Video:", video_path)
        print("=" * 60)

        try:

            with open(video_path, "rb") as video_file:

                response = requests.post(

                    self.webhook_url,

                    data={
                        "content": message
                    },

                    files={
                        "file": (
                            video_path.name,
                            video_file,
                            "video/mp4"
                        )
                    },

                    timeout=self.timeout

                )

            if response.status_code in (200, 204):

                print("Discord Upload Successfully.")

                return True

            else:

                print("Discord Upload Failed.")

                print(
                    "Status Code:",
                    response.status_code
                )

                print("Response:", response.text)

                return False

        except requests.exceptions.Timeout:

            print("Discord Upload Timeout.")

            return False

        except requests.exceptions.RequestException:

            print("Discord Network Request Failed.")

            return False

        except OSError:

            print("Cannot Read Video File.")

            return False


# =========================================================
# 4. 獨立初始化測試
# =========================================================

if __name__ == "__main__":

    uploader = DiscordUploader()

    print("Initialization Complete.")
