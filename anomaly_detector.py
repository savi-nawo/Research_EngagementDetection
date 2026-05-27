import time
import queue
import tkinter as tk
from tkinter import messagebox
from datetime import datetime, timedelta
from pymongo import MongoClient
from collections import defaultdict

# ==============================
# CONFIG
# ==============================
RUN_SPEED_TH = 50.0
LOITER_TIME_TH = 30.0
LOOK_RATIO_TH = 0.9

SCAN_INTERVAL = 1
ACTIVE_WINDOW = 3        # seconds → who is "currently visible"
ALERT_COOLDOWN = 60
CONFIRM_FRAMES = 30      # 🔥 trigger only after 30 detections

# ==============================
# MONGODB
# ==============================
client = MongoClient("mongodb://localhost:27017")
db = client.mall_analytics

people_col = db.engagement_logs
anomaly_col = db.anomaly_events

print("✓ MongoDB connected")

# ==============================
# THREAD-SAFE QUEUE
# ==============================
alert_queue = queue.Queue()

# ==============================
# DEDUP + CONFIRMATION STATE
# ==============================
recent_alerts = {}                     # cooldown
violation_counter = defaultdict(int)   # per ID + anomaly

# ==============================
# COOLDOWN CHECK
# ==============================
def can_trigger(key):
    now = datetime.utcnow()
    last = recent_alerts.get(key)

    if last is None or (now - last).total_seconds() > ALERT_COOLDOWN:
        recent_alerts[key] = now
        return True
    return False

# ==============================
# LOG + ALERT (AFTER CONFIRMATION)
# ==============================
def log_anomaly(track_id, anomaly_type, metric, threshold):
    confirm_key = f"{track_id}_{anomaly_type}"
    violation_counter[confirm_key] += 1

    # 🔒 Wait until confirmed enough times
    if violation_counter[confirm_key] < CONFIRM_FRAMES:
        return

    if not can_trigger(confirm_key):
        return

    anomaly_col.insert_one({
        "track_id": track_id,
        "anomaly_type": anomaly_type,
        "metric_value": metric,
        "threshold": threshold,
        "timestamp": datetime.utcnow()
    })

    msg = f"ID {track_id}\nAnomaly: {anomaly_type.upper()}"
    print(f"🚨 SECURITY ALERT | {msg}")
    alert_queue.put(msg)

# ==============================
# ANOMALY RULES
# ==============================
def detect_running(person):
    if person.get("avg_speed", 0) > RUN_SPEED_TH:
        log_anomaly(person["track_id"], "running",
                    person["avg_speed"], RUN_SPEED_TH)

def detect_loitering(person):
    if person.get("total_dwell", 0) > LOITER_TIME_TH:
        log_anomaly(person["track_id"], "loitering",
                    person["total_dwell"], LOITER_TIME_TH)

def detect_suspicious_idle(person):
    if person.get("total_dwell", 0) > 30 and person.get("looking_ratio", 1.0) < LOOK_RATIO_TH:
        log_anomaly(person["track_id"], "suspicious_idle_behavior",
                    person["looking_ratio"], LOOK_RATIO_TH)

# ==============================
# BACKGROUND DETECTOR LOOP
# ==============================
def anomaly_loop():
    print("🚨 Anomaly detection service started")

    while True:
        try:
            now = datetime.utcnow()
            active_since = now - timedelta(seconds=ACTIVE_WINDOW)

            # 🔥 ONLY currently visible people
            active_people = people_col.find({
                "last_seen": {"$gte": active_since}
            })

            for person in active_people:
                detect_running(person)
                detect_loitering(person)
                detect_suspicious_idle(person)

            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            print("❌ Anomaly error:", e)
            time.sleep(5)

# ==============================
# TKINTER UI (MAIN THREAD)
# ==============================
def ui_loop():
    root = tk.Tk()
    root.withdraw()

    def check_alerts():
        while not alert_queue.empty():
            msg = alert_queue.get()
            messagebox.showwarning("SECURITY ALERT", msg)

        root.after(500, check_alerts)

    root.after(500, check_alerts)
    root.mainloop()

# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    import threading

    threading.Thread(target=anomaly_loop, daemon=True).start()
    ui_loop()
