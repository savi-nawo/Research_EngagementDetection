from pymongo import MongoClient
import pandas as pd
import numpy as np

# ==============================
# DATABASE CONNECTION
# ==============================
client = MongoClient("mongodb://localhost:27017/")
db = client["mall_analytics"]
col = db["engagement_logs"]

print("✓ Connected to MongoDB")

# ==============================
# LOAD COMPLETED SESSIONS
# ==============================
docs = list(col.find(
    {"active": False},
    {
        "_id": 0,
        "looking_ratio": 1,
        "total_dwell": 1,
        "avg_speed": 1,
        "engaged": 1
    }
))

if not docs:
    raise RuntimeError("No completed sessions found.")

df = pd.DataFrame(docs)

#Split engaged vs non-engaged populations
df_engaged = df[df["engaged"] == True]
df_not_engaged = df[df["engaged"] == False]

#Learn adaptive thresholds (statistical)
LOOKING_RATIO_TH = (
    df_engaged["looking_ratio"].mean()
    - df_engaged["looking_ratio"].std()
)

DWELL_TIME_TH = (
    df_engaged["total_dwell"].mean()
    - df_engaged["total_dwell"].std()
)


SPEED_TH = (
    df_engaged["avg_speed"].mean()
    + df_engaged["avg_speed"].std()
)

LOOKING_RATIO_TH = float(np.clip(LOOKING_RATIO_TH, 0.2, 0.9))
DWELL_TIME_TH = float(max(DWELL_TIME_TH, 0.5))
SPEED_TH = float(min(SPEED_TH, 10.0))

#Output learned thresholds
print("\n=== 🔧 ADAPTIVE THRESHOLDS (LEARNED) ===")
print(f"Looking Ratio Threshold : {LOOKING_RATIO_TH:.3f}")
print(f"Dwell Time Threshold   : {DWELL_TIME_TH:.2f} sec")
print(f"Speed Upper Bound      : {SPEED_TH:.2f} px/frame")

#Save thresholds for reuse

thresholds = {
    "looking_ratio_th": LOOKING_RATIO_TH,
    "dwell_time_th": DWELL_TIME_TH,
    "speed_th": SPEED_TH
}

pd.Series(thresholds).to_json("adaptive_thresholds.json", indent=2)
print("\n✓ Thresholds saved to adaptive_thresholds.json")
