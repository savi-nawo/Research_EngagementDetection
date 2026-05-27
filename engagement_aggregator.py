import time
from datetime import datetime, timedelta
from pymongo import MongoClient

# ==============================
# CONFIG
# ==============================
WINDOW_SECONDS = 10
SCAN_INTERVAL = 2
ENGAGEMENT_THRESHOLD = 40.0
GRACE_SECONDS = 5

# ==============================
# MONGODB
# ==============================
client = MongoClient("mongodb://localhost:27017/")
db = client.mall_analytics

logs_col = db.engagement_logs
overall_col = db.overall_engagement
ad_col = db.ad_triggers

print("✓ Engagement aggregation service started")


# STATE
# ==============================
last_insert_time = None

# ==============================
# AD TRIGGER
# ==============================
def trigger_ad(engagement_pct, engaged, total):
    ad_col.insert_one({
        "timestamp": datetime.utcnow(),
        "engagement_percentage": engagement_pct,
        "engaged_people": engaged,
        "total_people": total,
        "action": "PLAY_AD"
    })

    print(
        f"📺 AD TRIGGERED | "
        f"Engagement={engagement_pct:.1f}% "
        f"({engaged}/{total})"
    )

# ==============================
# MAIN LOOP
# ==============================
while True:
    try:
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=WINDOW_SECONDS + GRACE_SECONDS)

        # 🔥 ONLY people currently in front of the screen
        people = list(
            logs_col.find({
                "last_seen": {"$gte": window_start},
                "active": True
            })
        )

        total_people = len(people)
        engaged_people = sum(1 for p in people if p.get("engaged", False))

        engagement_pct = (
            (engaged_people / total_people) * 100
            if total_people > 0 else 0
        )

        if (
            last_insert_time is None or
            (now - last_insert_time).total_seconds() >= WINDOW_SECONDS
        ):
            overall_col.insert_one({
                "window_start": window_start,
                "window_end": now,
                "total_people": total_people,
                "engaged_people": engaged_people,
                "engagement_percentage": engagement_pct,
                "timestamp": now
            })

            print(
                f"📊 Engagement (10s window) | "
                f"Total={total_people} "
                f"Engaged={engaged_people} "
                f"({engagement_pct:.1f}%)"
            )

            last_insert_time = now

            if engagement_pct >= ENGAGEMENT_THRESHOLD and total_people > 0:
                trigger_ad(engagement_pct, engaged_people, total_people)

        time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        print("\n🛑 Engagement aggregator stopped manually")
        break

    except Exception as e:
        print("❌ Aggregation error:", e)
        time.sleep(5)
