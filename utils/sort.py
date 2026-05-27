# utils/sort.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
import cv2


# =========================
# KALMAN TRACKER
# =========================
class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        # bbox = [x1,y1,x2,y2]

        self.box = bbox
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.hits = 0
        self.no_losses = 0

        # for drawing path
        self.trace = deque(maxlen=20)

    def update(self, bbox):
        """Update with new detection"""
        self.box = bbox
        self.hits += 1
        self.no_losses = 0

        # save center for trajectory
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        self.trace.append((cx, cy))

    def predict(self):
        """Predict next position (simplified for our project)"""
        self.no_losses += 1
        return self.box


# =========================
# SORT TRACKER
# =========================
class Sort:
    def __init__(self, max_age=10, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers = []

    def update(self, detections):
        # detections = Nx4 (x1,y1,x2,y2)
        detections = np.array(detections)

        if len(self.trackers) == 0:
            for det in detections:
                self.trackers.append(KalmanBoxTracker(det))
            return []

        # Step 1: Predict trackers
        predictions = []
        for t in self.trackers:
            predictions.append(t.predict())
        predictions = np.array(predictions)

        # Step 2: IoU matching
        iou_matrix = np.zeros((len(predictions), len(detections)), dtype=np.float32)

        for t_idx, trk in enumerate(predictions):
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = iou(trk, det)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched, unmatched_trackers, unmatched_detections = [], [], []

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] < self.iou_threshold:
                unmatched_trackers.append(r)
                unmatched_detections.append(c)
            else:
                matched.append((r, c))

        for t_idx in range(len(predictions)):
            if t_idx not in row_ind:
                unmatched_trackers.append(t_idx)

        for d_idx in range(len(detections)):
            if d_idx not in col_ind:
                unmatched_detections.append(d_idx)

        # Step 3: Update matched trackers
        for t_idx, d_idx in matched:
            self.trackers[t_idx].update(detections[d_idx])

        # Step 4: Create new trackers for unmatched detections
        for d_idx in unmatched_detections:
            self.trackers.append(KalmanBoxTracker(detections[d_idx]))

        # Step 5: Remove dead trackers
        alive_trackers = []
        for t in self.trackers:
            if t.no_losses < self.max_age:
                alive_trackers.append(t)
        self.trackers = alive_trackers

        # Step 6: Output final tracks
        outputs = []
        for t in self.trackers:
            if t.hits >= self.min_hits:
                outputs.append([t.box[0], t.box[1], t.box[2], t.box[3], t.id, list(t.trace)])

        return outputs


# =========================
# Helper – IoU
# =========================
def iou(bb1, bb2):
    xA = max(bb1[0], bb2[0])
    yA = max(bb1[1], bb2[1])
    xB = min(bb1[2], bb2[2])
    yB = min(bb1[3], bb2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    return inter / (area1 + area2 - inter + 1e-6)
