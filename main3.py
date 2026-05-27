import cv2
import torch
import time
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

from model.head_orientation_cnn import HeadOrientationCNN
from utils.sort import Sort
from mongo_router import create_person, update_person, finalize_person

from enhancements import ZoneEngine, zone_adjust_engagement

# ==============================
# CONFIG
# ==============================
CONF_TH = 0.35
MIN_MOTION_PX = 2.5

MAX_AGE = 20
MIN_HITS = 3
IOU_TH = 0.3

FPS_ASSUMED = 30
ENGAGE_TIME = 1.5
MIN_LOOK_FRAMES = int(FPS_ASSUMED * ENGAGE_TIME)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# LOAD MODELS
# ==============================
print("Loading Mall detector...")
detector = YOLO("weights/yolov8n.pt")
print("✓ Mall detector loaded")

head_model = HeadOrientationCNN().to(DEVICE)
head_model.load_state_dict(torch.load("weights/head_orientation_cnn.pt", map_location=DEVICE))
head_model.eval()
print("✓ Head orientation model loaded")

# ==============================
# SORT TRACKER
# ==============================
tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_TH)

# ==============================
# ZONE ENGINE
# ==============================
zone_engine = ZoneEngine("zones_cctv.json")

# ==============================
# HEAD CROP
# ==============================
def crop_head(frame, bbox):
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    head_y2 = y1 + int(0.4 * h)
    head = frame[y1:head_y2, x1:x2]
    if head.size == 0:
        return None
    return cv2.resize(head, (64, 64))

# ==============================
# HEAD ORIENTATION
# ==============================
def infer_head_orientation(head_crop):
    t = torch.tensor(head_crop / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        out = head_model(t)
    return "looking_at_kiosk" if out.argmax().item() == 1 else "looking_away"

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)
print("🎥 Webcam started — Press Q to quit")

motion_history = defaultdict(lambda: deque(maxlen=5))
person_db_state = {}
prev_time = time.time()

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ==============================
    # DEBUG: DRAW ZONES (TEMPORARY)
    # ==============================
    for zone_name, poly in zone_engine.zones.items():
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

        cx_z = int(np.mean([p[0] for p in poly]))
        cy_z = int(np.mean([p[1] for p in poly]))
        cv2.putText(frame, zone_name, (cx_z, cy_z),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # -------- DETECTION --------
    results = detector(frame, conf=CONF_TH, iou=0.5, classes=[0], verbose=False)[0]
    detections = []

    for b in results.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        if (x2 - x1) < 40 or (y2 - y1) < 60:
            continue
        detections.append([x1, y1, x2, y2, float(b.conf)])

    dets_np = np.array(detections) if detections else np.empty((0, 5))
    tracks = tracker.update(dets_np)

    active_tracks = 0

    for tr in tracks:
        x1, y1, x2, y2, tid = map(int, tr[:5])

        if tid not in person_db_state:
            person_db_state[tid] = create_person(tid)
            person_db_state[tid].update({
                "looking_frames": 0,
                "total_frames": 0,
                "engaged": False,
                "zone": "unknown"
            })

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        zone = zone_engine.get_zone(cx, cy)

        motion_history[tid].append((cx, cy))

        motion = 0.0
        if len(motion_history[tid]) >= 2:
            xa, ya = motion_history[tid][-2]
            xb, yb = motion_history[tid][-1]
            motion = float(np.hypot(xb - xa, yb - ya))

        head = crop_head(frame, (x1, y1, x2, y2))
        label = "NO_HEAD"

        if head is not None:
            o = infer_head_orientation(head)
            person_db_state[tid]["total_frames"] += 1

            if o == "looking_at_kiosk":
                person_db_state[tid]["looking_frames"] += 1
                label = "LOOKING"
            else:
                label = "AWAY"

        # --------------------------
        # BASE ENGAGEMENT LOGIC (UNCHANGED)
        # --------------------------
        base_engaged = person_db_state[tid]["looking_frames"] >= MIN_LOOK_FRAMES

        # Zone-aware interpretation
        final_engaged = zone_adjust_engagement(
            base_engaged,
            zone,
            person_db_state[tid]["total_frames"]
        )

        # ✅ SET STATE FIRST (IMPORTANT)
        person_db_state[tid]["engaged"] = final_engaged
        person_db_state[tid]["zone"] = zone

        # ✅ THEN UPDATE DB ONCE
        fps = 1.0 / max(time.time() - prev_time, 1e-6)
        update_person(person_db_state[tid], motion, fps)

        active_tracks += 1
        color = (0, 255, 0) if motion >= MIN_MOTION_PX else (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {tid} | {zone} | mv:{motion:.1f} | {label}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------- FINALIZE PEOPLE WHO LEFT --------
    current_ids = set(int(t[4]) for t in tracks)
    for tid in list(person_db_state.keys()):
        if tid not in current_ids:
            s = person_db_state[tid]
            finalize_person({
                **s,
                "looking_ratio": s["looking_frames"] / max(s["total_frames"], 1),
                "engaged": s["engaged"],
                "zone": s.get("zone", "unknown")
            })
            del person_db_state[tid]

    fps = 1.0 / max(time.time() - prev_time, 1e-6)
    prev_time = time.time()

    cv2.putText(frame, f"FPS:{fps:.1f} | Active:{active_tracks}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Mall Analytics", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------- FINALIZE ALL --------
for tid, s in person_db_state.items():
    finalize_person({
        **s,
        "looking_ratio": s["looking_frames"] / max(s["total_frames"], 1),
        "engaged": s["engaged"],
        "zone": s.get("zone", "unknown")
    })

cap.release()
cv2.destroyAllWindows()
print("✓ All active persons finalized")
