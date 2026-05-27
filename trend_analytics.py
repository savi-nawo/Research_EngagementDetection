from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# DATABASE CONNECTION
# ==============================
client = MongoClient("mongodb://localhost:27017/")
db = client["mall_analytics"]
col = db["engagement_logs"]

print("✓ Connected to MongoDB")

# ==============================
# LOAD DATA FROM MONGODB
# ==============================
docs = list(col.find(
    {"active": False},
    {
        "_id": 0,
        "track_id": 1,
        "first_seen": 1,
        "last_seen": 1,
        "total_dwell": 1,
        "looking_ratio": 1,
        "engaged": 1,
        "zone": 1
    }
))

if not docs:
    print("No completed engagement sessions found.")
    exit()

df = pd.DataFrame(docs)

# ==============================
# TIMEZONE CONVERSION
# ==============================
df["ts_utc"] = pd.to_datetime(df["first_seen"], utc=True)
df["ts_local"] = df["ts_utc"].dt.tz_convert("Asia/Colombo")

# ==============================
# TIME FEATURES
# ==============================
df["hour"] = df["ts_local"].dt.hour
df["day_name"] = df["ts_local"].dt.day_name()
df["date"] = df["ts_local"].dt.date

# ==============================
# TRUE ENGAGEMENT METRIC
# ==============================
df["engaged_flag"] = df["engaged"].astype(int)

# ==============================
# 1️⃣ HOURLY ENGAGEMENT RATE
# ==============================
hourly_stats = (
    df.groupby("hour")
    .agg(
        total_people=("track_id", "count"),
        engaged_people=("engaged_flag", "sum")
    )
    .reset_index()
    .sort_values("hour")
)

hourly_stats["engagement_rate"] = (
    hourly_stats["engaged_people"] / hourly_stats["total_people"]
)

hourly_stats["hour_range"] = (
    hourly_stats["hour"].astype(str) + ":00–" +
    (hourly_stats["hour"] + 1).astype(str) + ":00"
)

print("\n=== HOURLY TRUE ENGAGEMENT RATE ===")
print(hourly_stats[["hour_range", "total_people", "engaged_people", "engagement_rate"]])

# ==============================
# 2️⃣ DAY-OF-WEEK ENGAGEMENT RATE
# ==============================
dow_stats = (
    df.groupby("day_name")
    .agg(
        total_people=("track_id", "count"),
        engaged_people=("engaged_flag", "sum")
    )
    .reset_index()
)

dow_stats["engagement_rate"] = (
    dow_stats["engaged_people"] / dow_stats["total_people"]
)

dow_stats = dow_stats.sort_values("engagement_rate", ascending=False)

print("\n=== DAY-OF-WEEK TRUE ENGAGEMENT RATE ===")
print(dow_stats)

# ==============================
# 3️⃣ ZONE-WISE ENGAGEMENT RATE
# ==============================
zone_stats = (
    df.groupby("zone")
    .agg(
        total_people=("track_id", "count"),
        engaged_people=("engaged_flag", "sum")
    )
    .reset_index()
)

zone_stats["engagement_rate"] = (
    zone_stats["engaged_people"] / zone_stats["total_people"]
)

zone_stats = zone_stats.sort_values("engagement_rate", ascending=False)

print("\n=== ZONE-WISE TRUE ENGAGEMENT RATE ===")
print(zone_stats)

# ==============================
# FINAL PEAK INSIGHTS
# ==============================
peak_hour = hourly_stats.loc[hourly_stats["engagement_rate"].idxmax()]
peak_zone = zone_stats.loc[zone_stats["engagement_rate"].idxmax()]

print("\n🔥 FINAL INSIGHTS")
print(f"Peak Engagement Hour: {peak_hour['hour_range']}")
print(f"Engagement Rate: {peak_hour['engagement_rate']:.2%}")
print(f"Best Performing Zone: {peak_zone['zone']}")
print(f"Engagement Rate: {peak_zone['engagement_rate']:.2%}")

# ==============================
# 📊 VISUALIZATIONS (BAR CHARTS)
# ==============================

# ---- Hourly Engagement Rate Chart ----
plt.figure()
plt.bar(hourly_stats["hour_range"], hourly_stats["engagement_rate"])
plt.xlabel("Hour Range")
plt.ylabel("Engagement Rate")
plt.title("Hourly Engagement Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---- Day-of-Week Engagement Rate Chart ----
plt.figure()
plt.bar(dow_stats["day_name"], dow_stats["engagement_rate"])
plt.xlabel("Day")
plt.ylabel("Engagement Rate")
plt.title("Day-of-Week Engagement Rate")
plt.tight_layout()
plt.show()

# ---- Zone Engagement Rate Chart ----
plt.figure()
plt.bar(zone_stats["zone"], zone_stats["engagement_rate"])
plt.xlabel("Zone")
plt.ylabel("Engagement Rate")
plt.title("Zone-wise Engagement Rate")
plt.tight_layout()
plt.show()
