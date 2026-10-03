import argparse
import json
import mimetypes
import os
import time
from pathlib import Path

import requests


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
STABLE_SECONDS = 10
RETRY_SECONDS = 30
MAX_ATTEMPTS = 3


def send_discord(message, video_path=None):
    """傳送訊息，可附加一支已完成儲存的影片。"""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

    if not webhook_url:
        print("❌ 請先設定 DISCORD_WEBHOOK_URL", flush=True)
        return False

    payload = {
        "content": message,
        "allowed_mentions": {"parse": []},
    }

    try:
        if video_path is None:
            response = requests.post(
                webhook_url,
                params={"wait": "true"},
                json=payload,
                timeout=(10, 30),
            )
        else:
            path = Path(video_path).expanduser().resolve()

            if not path.is_file() or path.stat().st_size == 0:
                print(f"❌ 影片不存在或為空檔案：{path}", flush=True)
                return False

            content_type = (
                mimetypes.guess_type(path.name)[0]
                or "application/octet-stream"
            )

            with path.open("rb") as video:
                response = requests.post(
                    webhook_url,
                    params={"wait": "true"},
                    data={
                        "payload_json": json.dumps(
                            payload, ensure_ascii=False
                        )
                    },
                    files={
                        "files[0]": (
                            path.name,
                            video,
                            content_type,
                        )
                    },
                    timeout=(10, 120),
                )

        if response.status_code == 200:
            print("✅ Discord 傳送成功", flush=True)
            return True

        print(
            f"❌ 傳送失敗，HTTP {response.status_code}",
            flush=True,
        )

        if response.status_code == 429:
            print("傳送過於頻繁，請稍後再試。", flush=True)
        elif response.status_code == 413:
            print("影片太大，請壓縮或縮短影片。", flush=True)
        elif response.status_code in (401, 403, 404):
            print("請確認 Webhook 是否有效及可用。", flush=True)

        return False

    except requests.RequestException:
        # 不印出例外內容，避免洩漏 Webhook token
        print(
            "❌ 網路連線或傳送逾時，請確認 Discord 是否已收到。",
            flush=True,
        )
        return False
    except OSError:
        print("❌ 無法讀取影片檔案。", flush=True)
        return False


def list_videos(folder):
    """只讀取指定資料夾內的影片，不包含子資料夾。"""
    return {
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    }


def watch_video_folders(fall_dir=None, violence_dir=None):
    """監看一個或兩個資料夾，傳送啟動後出現的新影片。"""
    folders = {}

    if fall_dir:
        folders["跌倒"] = Path(fall_dir).expanduser().resolve()

    if violence_dir:
        folders["疑似暴力行為"] = (
            Path(violence_dir).expanduser().resolve()
        )

    if not folders:
        raise ValueError("至少需要提供一個影片資料夾。")

    if len(set(folders.values())) != len(folders):
        raise ValueError("跌倒與暴力影片不能設定為同一個資料夾。")

    known = set()
    pending = {}

    for event, folder in folders.items():
        if not folder.is_dir():
            raise FileNotFoundError(
                f"{event}影片資料夾不存在：{folder}"
            )

        known.update(list_videos(folder))
        print(f"👀 監看{event}影片：{folder}", flush=True)

    print(
        f"已略過 {len(known)} 支既有影片。\n"
        "請保持這個終端機執行，再啟動偵測程式。\n"
        f"新影片連續 {STABLE_SECONDS} 秒沒有變動後會嘗試傳送。\n"
        "按 Ctrl+C 停止監看。",
        flush=True,
    )

    try:
        while True:
            # 尋找新影片
            for event, folder in folders.items():
                try:
                    videos = list_videos(folder)
                except OSError:
                    print(f"⚠️ 暫時無法讀取：{folder}", flush=True)
                    continue

                for path in sorted(videos - known):
                    known.add(path)
                    pending[path] = {
                        "event": event,
                        "signature": None,
                        "stable_since": time.monotonic(),
                        "retry_at": 0,
                        "attempts": 0,
                    }
                    print(
                        f"📹 發現新{event}影片：{path.name}",
                        flush=True,
                    )

            # 等待影片檔案停止變動，再上傳
            for path, item in list(pending.items()):
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    print(f"⚠️ 影片已移除：{path}", flush=True)
                    del pending[path]
                    continue
                except OSError:
                    continue

                now = time.monotonic()
                signature = (stat.st_size, stat.st_mtime_ns)

                if signature != item["signature"]:
                    item["signature"] = signature
                    item["stable_since"] = now
                    continue

                if (
                    stat.st_size == 0
                    or now - item["stable_since"] < STABLE_SECONDS
                    or now < item["retry_at"]
                ):
                    continue

                message = (
                    f"🚨 偵測到{item['event']}事件\n"
                    f"事件影片：{path.name}\n"
                    "請查看附件確認現場狀況。"
                )

                print(f"⬆️ 正在傳送：{path.name}", flush=True)

                if send_discord(message, video_path=path):
                    del pending[path]
                else:
                    item["attempts"] += 1

                    if item["attempts"] >= MAX_ATTEMPTS:
                        print(
                            f"❌ 已嘗試 {MAX_ATTEMPTS} 次，"
                            f"請手動重傳：{path}",
                            flush=True,
                        )
                        del pending[path]
                    else:
                        item["retry_at"] = (
                            time.monotonic() + RETRY_SECONDS
                        )
                        print(
                            f"⏳ {RETRY_SECONDS} 秒後重新嘗試。",
                            flush=True,
                        )

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n已停止監看。", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Discord 訊息與事件影片傳送程式"
    )
    parser.add_argument("--video", help="手動傳送指定影片")
    parser.add_argument("--fall-dir", help="跌倒影片資料夾")
    parser.add_argument("--violence-dir", help="暴力影片資料夾")
    args = parser.parse_args()

    if args.video and (args.fall_dir or args.violence_dir):
        parser.error("--video 不能與資料夾監看參數一起使用。")

    if not args.video and not (args.fall_dir or args.violence_dir):
        parser.error("請提供 --video、--fall-dir 或 --violence-dir。")

    if not os.getenv("DISCORD_WEBHOOK_URL", "").strip():
        parser.error("請先設定 DISCORD_WEBHOOK_URL。")

    if args.video:
        success = send_discord(
            "🧪 專題傳輸測試：請查看附件影片。",
            video_path=args.video,
        )
        return 0 if success else 1

    try:
        watch_video_folders(args.fall_dir, args.violence_dir)
    except (OSError, ValueError) as error:
        print(f"❌ {error}", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())