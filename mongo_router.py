from pymongo import MongoClient
from datetime import datetime

# ==============================
# CONNECTION
# ==============================
client = MongoClient("mongodb://localhost:27017/")
db = client["mall_analytics"]

# Live analytics (used by engagement aggregator + anomaly detector)
analytics_col = db["engagement_logs"]

# Standardized integration feed (used by fusion service / recommender)
integration_col = db["detections_raw"]

print("✓ MongoDB connected")

# ==============================
# PERSON LIFECYCLE FUNCTIONS
# ==============================

def create_person(track_id, session_id="default_session"):
    """
    Create a LIVE person record as soon as a track appears.
    """
    state = {
        "track_id": track_id,
        "session_id": session_id,

        "first_seen": datetime.utcnow(),
        "last_seen": datetime.utcnow(),

        "frames_seen": 0,
        "total_dwell": 0.0,
        "avg_speed": 0.0,

        # engagement signals
        "looking_frames": 0,
        "total_frames": 0,
        "looking_ratio": 0.0,
        "engaged": False,

        "zone": "unknown",

        "active": True
    }

    # 🔹 Insert immediately so the person is visible to aggregators
    analytics_col.insert_one(state)

    return state


def update_person(state, motion, fps):
    """
    Update LIVE engagement statistics every frame.
    """
    state["frames_seen"] += 1
    state["last_seen"] = datetime.utcnow()

    if motion > 0:
        state["total_dwell"] += 1.0 / fps

    # running average speed
    state["avg_speed"] = (
        (state["avg_speed"] * (state["frames_seen"] - 1) + motion)
        / state["frames_seen"]
    )

    # update looking ratio continuously
    state["looking_ratio"] = (
        state.get("looking_frames", 0)
        / max(state.get("total_frames", 1), 1)
    )

    analytics_col.update_one(
        {"track_id": state["track_id"]},
        {"$set": {
            "last_seen": state["last_seen"],
            "frames_seen": state["frames_seen"],
            "total_dwell": state["total_dwell"],
            "avg_speed": state["avg_speed"],

            "looking_frames": state["looking_frames"],
            "total_frames": state["total_frames"],
            "looking_ratio": state["looking_ratio"],
            "engaged": state["engaged"],

            "zone": state.get("zone", "unknown"),

            "active": True
        }},
        upsert=True
    )


def finalize_person(state):
    """
    Called once when a person leaves the scene.
    Marks them inactive and emits standardized integration output.
    """
    state["active"] = False

    # 🔹 Mark inactive (do NOT create a new document)
    analytics_col.update_one(
        {"track_id": state["track_id"]},
        {"$set": {
            "active": False,
            "last_seen": datetime.utcnow(),
            "engaged": state.get("engaged", False),
            "looking_ratio": state.get("looking_ratio", 0.0),

            "zone": state.get("zone", "unknown") 

        }}
    )

    # 🔹 Write standardized behaviour output for integration
    integration_col.insert_one({
        "model": "behaviour",

        "session_id": state.get("session_id", "default_session"),
        "person_id": f"track_{state['track_id']}",
        "timestamp": datetime.utcnow(),

        "age_group": None,
        "gender": None,
        "emotion": None,
        "behaviour": "engaged" if state.get("engaged") else "not_engaged",

        "confidence": {
            "age_gender": None,
            "emotion": None,
            "behaviour": state.get("looking_ratio", 0.0)
        }
    })

    print(f"📥 MongoDB FINALIZED | track_id={state['track_id']}")
