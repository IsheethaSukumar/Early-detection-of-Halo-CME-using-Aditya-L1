"""
Phase 7 & 13: Alert Manager for Telegram and Slack Notifications
===================================================================
Handles real-time alerts for CRITICAL and WARNING CME risk events,
with deduplication / cooldown logic to prevent notification spam.
"""

import os
import time
import requests
from typing import Dict, Any, Optional

# Load threshold & credentials from env
WARNING_THRESHOLD = float(os.getenv("WARNING_THRESHOLD", "0.37"))
CRITICAL_THRESHOLD = float(os.getenv("CRITICAL_THRESHOLD", "0.85"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# Cooldown in seconds (e.g., 30 minutes = 1800s)
COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "1800"))


class AlertManager:
    def __init__(self):
        self.last_alert_time = {}

    def _should_send(self, alert_key: str) -> bool:
        now = time.time()
        last = self.last_alert_time.get(alert_key, 0)
        if now - last >= COOLDOWN_SECONDS:
            self.last_alert_time[alert_key] = now
            return True
        return False

    def send_telegram_alert(self, message: str) -> bool:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("[AlertManager] Telegram credentials missing, skipping telegram notification.")
            return False
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            resp = requests.post(url, json=payload, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            print(f"[AlertManager] Telegram send error: {e}")
            return False

    def send_slack_alert(self, message: str) -> bool:
        if not SLACK_WEBHOOK_URL:
            print("[AlertManager] Slack webhook URL missing, skipping slack notification.")
            return False
        payload = {"text": message}
        try:
            resp = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            print(f"[AlertManager] Slack send error: {e}")
            return False

    def process_prediction(self, prediction_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        risk = prediction_data.get("risk_level", "NORMAL")
        prob = prediction_data.get("cme_probability", 0.0)
        ts = prediction_data.get("timestamp", "N/A")
        is_anomaly = prediction_data.get("is_anomaly", False)

        if risk not in ["WARNING", "CRITICAL"]:
            return None

        alert_key = f"{risk}"
        if not self._should_send(alert_key):
            print(f"[AlertManager] Cooldown active for {risk} alerts. Suppressing duplicate alert.")
            return None

        status_emoji = "🔴" if risk == "CRITICAL" else "⚠️"
        msg = (
            f"{status_emoji} *Aditya-L1 Space Weather Alert*\n"
            f"*Risk Level:* {risk}\n"
            f"*CME Probability:* {prob:.4f}\n"
            f"*Timestamp:* {ts}\n"
            f"*Sensor Anomaly Flag:* {is_anomaly}\n"
            f"_Notification from Near-Real-Time SWIS Pipeline_"
        )

        sent_tg = self.send_telegram_alert(msg)
        sent_slack = self.send_slack_alert(msg)

        return {
            "timestamp": ts,
            "alert_type": risk,
            "probability": prob,
            "message": msg,
            "status": "SENT" if (sent_tg or sent_slack) else "DISPATCH_FAILED_OR_NO_CREDENTIALS"
        }
