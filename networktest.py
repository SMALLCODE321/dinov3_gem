import requests
import schedule
import time
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# ===== 配置 =====
PROXY = "http://127.0.0.1:7897"
URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/19/202613/105377"

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465

SENDER_EMAIL = "xuliujia3@gmail.com"
SENDER_PASSWORD = "ewkp hnfx fixm qlwh"   # ⚠️填16位应用密码
RECEIVER_EMAIL = "3501990698@qq.com"


def send_email(content):
    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = "ArcGIS 监控结果"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        print("📧 邮件发送成功")
    except Exception as e:
        print("❌ 邮件发送失败:", e)


def check_server():
    print("⏰ 检测:", datetime.now())

    proxies = {
        "http": PROXY,
        "https": PROXY,
    }

    try:
        r = requests.head(URL, proxies=proxies, timeout=20)

        content = f"""
时间: {datetime.now()}
状态码: {r.status_code}
Headers:
{r.headers}
"""
        print(content)
        send_email(content)

    except Exception as e:
        err = f"请求失败: {e}"
        print(err)
        send_email(err)


# 每2小时执行
schedule.every(1).hours.do(check_server)

print("🚀 监控启动...")

check_server()

while True:
    schedule.run_pending()
    time.sleep(60)