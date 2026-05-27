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
        "track_id": 1,
        "total_dwell": 1,
        "avg_speed": 1,
        "looking_ratio": 1,
        "zone": 1,
        "engaged": 1
    }
))

if not docs:
    raise RuntimeError("No completed sessions found.")

df = pd.DataFrame(docs)
print(f"Loaded {len(df)} sessions")

#Encode zone (categorical → numeric) Clustering requires numeric vectors

zone_map = {z: i for i, z in enumerate(df["zone"].unique())}
df["zone_id"] = df["zone"].map(zone_map)

print("Zone encoding:", zone_map)

#Build feature vectors
# ==============================
# FEATURE VECTOR
# ==============================
X = df[
    ["total_dwell", "looking_ratio", "avg_speed", "zone_id"]
].fillna(0)

#Normalize features (mandatory)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Apply K-Means clustering
from sklearn.cluster import KMeans

K = 3
kmeans = KMeans(n_clusters=K, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["behavior_cluster"] = clusters

#Interpret clusters
cluster_summary = (
    df.groupby("behavior_cluster")
    .agg(
        avg_dwell=("total_dwell", "mean"),
        avg_speed=("avg_speed", "mean"),
        avg_looking_ratio=("looking_ratio", "mean"),
        engagement_rate=("engaged", "mean"),
        count=("track_id", "count")
    )
    .reset_index()
)

print("\n=== BEHAVIOR CLUSTER SUMMARY ===")
print(cluster_summary)

#Save cluster labels back to MongoDB
for _, row in df.iterrows():
    col.update_one(
        {"track_id": row["track_id"]},
        {"$set": {
            "behavior_cluster": int(row["behavior_cluster"])
        }}
    )

print("✓ Cluster labels saved to MongoDB")

