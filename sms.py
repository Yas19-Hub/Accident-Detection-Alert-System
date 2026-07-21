# sms_alert.py
import os
from datetime import datetime
from twilio.rest import Client
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# ====== CONFIG - FROM ENV ======
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")

FROM_PHONE  = os.getenv("TWILIO_FROM_PHONE", "(804) 636-2980")   # Twilio number
TO_HOSPITAL = os.getenv("ALERT_HOSPITAL_PHONE", "+9198457750270")
TO_POLICE   = os.getenv("ALERT_POLICE_PHONE", "+919845775027")

CAMERA_NAME     = "SGBIT Main Gate CCTV-01"
CAMERA_ADDRESS  = "S.G. Balekundri Institute of Technology, Belagavi, Karnataka"
CAMERA_GPS      = "15.8490, 74.4977"   # example coords

# ==========================================

def _build_message(video_path: str,
                   positive_prob: float,
                   first_accident_time: str | None = None) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    msg_lines = [
        "[ACCIDENT ALERT]",
        f"Detected at: {now_str}",
        f"Camera: {CAMERA_NAME}",
        f"Address: {CAMERA_ADDRESS}",
        f"GPS: {CAMERA_GPS}",
        "",
        f"Video file: {video_path}",
        f"Accident confidence: {positive_prob:.2f}",
    ]

    if first_accident_time is not None:
        msg_lines.append(f"First accident frame time in video: {first_accident_time}")

    msg_lines.append("")
    msg_lines.append("Please dispatch emergency response units ASAP.")

    return "\n".join(msg_lines)


def send_accident_alert(video_path: str,
                        positive_prob: float,
                        first_accident_time: str | None = None):
    """
    Send SMS to hospital + police when accident is detected.
    Includes DEBUG prints so you can see exactly what happens.
    """

    # ---- 1. Check credentials ----
    if not ACCOUNT_SID or not AUTH_TOKEN:
        raise RuntimeError(
            "Twilio credentials missing. "
            "Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in your .env file."
        )

    print("[DEBUG] Twilio ACCOUNT_SID loaded:", ACCOUNT_SID[:6] + "..." if ACCOUNT_SID else None)
    print("[DEBUG] FROM_PHONE:", FROM_PHONE)
    print("[DEBUG] TO_HOSPITAL:", TO_HOSPITAL)
    print("[DEBUG] TO_POLICE:", TO_POLICE)

    client = Client(ACCOUNT_SID, AUTH_TOKEN)

    body = _build_message(
        video_path=video_path,
        positive_prob=positive_prob,
        first_accident_time=first_accident_time,
    )

    # send to both with labels
    recipients = [
        ("Hospital", TO_HOSPITAL),
        ("Police", TO_POLICE),
    ]

    for label, to in recipients:
        if not to or to.strip() == "":
            print(f"[WARN] {label} phone is empty or not set. Skipping.")
            continue

        print(f"[DEBUG] Trying to send SMS to {label}: {to}")
        try:
            message = client.messages.create(
                body=body,
                from_=FROM_PHONE,
                to=to,
            )
            print(f"[SMS] Sent alert to {label} ({to}). SID={message.sid}")
        except Exception as e:
            print(f"[ERROR] Failed to send SMS to {label} ({to}): {e}")
