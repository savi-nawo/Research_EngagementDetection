import cv2
import torch
import json
import time
import numpy as np
from collections import defaultdict, deque

from model.malldetector import MallDetector
from model.head_orientation_cnn import HeadOrientationCNN
from scripts.decode_predictions_clean import decode_predictions_clean
from utils.sort import Sort

#  MongoDB router
from mongo_router import create_person, update_person, finalize_person

# ==============================
# CONFIG
# ============================== 
IMG_SIZE = 416
CONF_TH = 0.25
MIN_MOTION_PX = 2.5

MAX_AGE = 20
MIN_HITS = 3
IOU_TH = 0.3

FPS_ASSUMED = 30
ENGAGE_TIME = 1.5
MIN_LOOK_FRAMES = int(FPS_ASSUMED * ENGAGE_TIME)

# ==============================
# IOU (duplicate suppression)
# ==============================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter + 1e-6

    return inter / union

# ==============================
# LOAD ANCHORS
# ==============================
with open("scripts/anchors.json", "r") as f:
    anchors = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==============================
# LOAD DETECTOR
# ==============================
model = MallDetector(anchors).to(device)
model.load_state_dict(torch.load("weights/custom_best.pt", map_location=device))
model.eval()
print("✓ MallDetector loaded")

# ==============================
# LOAD HEAD ORIENTATION MODEL
# ==============================
head_model = HeadOrientationCNN().to(device)
head_model.load_state_dict(
    torch.load("weights/head_orientation_cnn.pt", map_location=device)
)
head_model.eval()
print("✓ Head orientation model loaded")

# ==============================
# SORT TRACKER
# ==============================
tracker = Sort(
    max_age=MAX_AGE,
    min_hits=MIN_HITS,
    iou_threshold=IOU_TH
)

# ==============================
# PREPROCESS
# ==============================
def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.tensor(img / 255.0).permute(2, 0, 1).float()
    return x.unsqueeze(0).to(device)

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
# HEAD ORIENTATION INFERENCE
# ==============================
def infer_head_orientation(head_crop):
    tensor = torch.tensor(head_crop / 255.0)\
        .permute(2, 0, 1)\
        .unsqueeze(0)\
        .float()\
        .to(device)

    with torch.no_grad():
        out = head_model(tensor)
    print("Head logits:", out.cpu().numpy())

    return "looking_at_kiosk" if out.argmax().item() == 1 else "looking_away"



    

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found")
    exit()

prev_time = time.time()
log_timer = time.time()

motion_history = defaultdict(lambda: deque(maxlen=5))
person_db_state = {}

print("🎥 Webcam started — Press Q to quit")

# ==============================
# MAIN LOOP
# ==============================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    inp = preprocess(frame)

    # ------------------------------
    # DETECTION
    # ------------------------------
    with torch.no_grad():
        outputs = model(inp)

    boxes, scores = decode_predictions_clean(
        outputs,
        anchors,
        conf_threshold=CONF_TH,
        nms_iou=0.5
    )

    detections = []

    if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
        for (x1, y1, x2, y2), sc in zip(
            boxes.cpu().numpy(),
            scores.cpu().numpy()
        ):
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)

            keep = True
            for d in detections:
                if iou(d[:4], [x1, y1, x2, y2]) > 0.5:
                    keep = False
                    break

            if keep:
                detections.append([x1, y1, x2, y2, sc])

    # ------------------------------
    # TRACKING
    # ------------------------------
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
                "engaged": False
            })

        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        motion_history[tid].append((cx, cy))

        motion = 0.0
        if len(motion_history[tid]) >= 2:
            (xa, ya), (xb, yb) = motion_history[tid][-2], motion_history[tid][-1]
            motion = np.hypot(xb - xa, yb - ya)

        fps = 1.0 / max(time.time() - prev_time, 1e-6)
        update_person(person_db_state[tid], motion, fps)

        # ------------------------------
        # HEAD ORIENTATION
        # ------------------------------
        head_crop = crop_head(frame, (x1, y1, x2, y2))
        label = "NO_HEAD"

        if head_crop is not None:
            orientation = infer_head_orientation(head_crop)
            person_db_state[tid]["total_frames"] += 1

            if orientation == "looking_at_kiosk":
                person_db_state[tid]["looking_frames"] += 1
                label = "LOOKING"
            else:
                label = "AWAY"

        if person_db_state[tid]["looking_frames"] >= MIN_LOOK_FRAMES:
            person_db_state[tid]["engaged"] = True

        active_tracks += 1
        color = (0,255,0) if motion >= MIN_MOTION_PX else (0,165,255)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(
            frame,
            f"ID {tid} | mv:{motion:.1f} | {label}",
            (x1, y1-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # ------------------------------
    # FINALIZE PEOPLE WHO LEFT
    # ------------------------------
    current_ids = set(int(t[4]) for t in tracks)

    for tid in list(person_db_state.keys()):
        if tid not in current_ids:
            state = person_db_state[tid]
            finalize_person({
                **state,
                "looking_ratio": state["looking_frames"] / max(state["total_frames"], 1),
                "engaged": state["engaged"]
            })
            del person_db_state[tid]

    # ------------------------------
    # FPS DISPLAY
    # ------------------------------
    now = time.time()
    fps = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | Active: {active_tracks}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,0),
        2
    )

    cv2.imshow("Mall Analytics = Detection + Tracking + Engagement", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # ==============================
# FINALIZE ALL ACTIVE PERSONS ON EXIT
# ==============================
for tid, state in person_db_state.items():
    finalize_person({
        **state,
        "looking_ratio": state["looking_frames"] / max(state["total_frames"], 1),
        "engaged": state["engaged"]
    })

print("✓ All active persons finalized")


cap.release()
cv2.destroyAllWindows()
