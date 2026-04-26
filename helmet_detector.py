"""
Helmet detection module — YOLOv8 + centroid tracking + multi-frame smoothing.

Overlap calculation
-------------------
We use simple bounding-box *intersection area* rather than IoU to decide
whether a helmet belongs to a person's head region.

    ix1 = max(head.x1, helmet.x1),  ix2 = min(head.x2, helmet.x2)
    iy1 = max(head.y1, helmet.y1),  iy2 = min(head.y2, helmet.y2)
    area = (ix2 - ix1) * (iy2 - iy1)  if ix2 > ix1 and iy2 > iy1 else 0

If area > 0 → helmet overlaps the head region → person is wearing a helmet.

IoU is deliberately avoided here: a helmet box is much smaller than the
head-region box, so the union dominates and IoU stays low even for a
perfect spatial match, making it a noisy threshold to tune.
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Tunable constants
# ──────────────────────────────────────────────────────────────────────────────

HEAD_FRACTION    = 0.25   # fraction of person height used as head region
MIN_HELMET_CONF  = 0.40   # minimum YOLO confidence for a helmet detection
HISTORY_LEN      = 5      # frames kept per person for majority-vote smoothing
TRACKER_MAX_DIST = 80     # max pixel distance for centroid-based ID reuse


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def head_region(person_box: tuple) -> tuple:
    """Top HEAD_FRACTION of a person bounding box, used as the head area."""
    x1, y1, x2, y2 = person_box
    head_h = max(1, int((y2 - y1) * HEAD_FRACTION))
    return (x1, y1, x2, y1 + head_h)


def intersection_area(a: tuple, b: tuple) -> int:
    """Axis-aligned bounding-box intersection area; 0 when boxes do not overlap."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def iou(a: tuple, b: tuple) -> float:
    """Intersection-over-Union of two axis-aligned boxes (provided for completeness)."""
    inter = intersection_area(a, b)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


# ──────────────────────────────────────────────────────────────────────────────
# Centroid Tracker
# ──────────────────────────────────────────────────────────────────────────────

class CentroidTracker:
    """
    Assigns stable integer IDs to bounding boxes across frames using
    greedy nearest-centroid matching.  IDs that vanish from a frame are
    immediately retired (no keep-alive buffer needed for this use-case).
    """

    def __init__(self, max_distance: int = TRACKER_MAX_DIST):
        self.max_distance = max_distance
        self.next_id      = 0
        self._objects     = {}   # id -> (centroid, box)

    @staticmethod
    def _centroid(box: tuple) -> tuple:
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, boxes: list) -> dict:
        """
        Match *boxes* to existing tracked IDs; mint new IDs for unmatched boxes.

        Returns:
            dict {id: (x1,y1,x2,y2)} containing only IDs visible this frame.
        """
        if not boxes:
            self._objects.clear()
            return {}

        new_centroids = [self._centroid(b) for b in boxes]

        if not self._objects:
            for c, b in zip(new_centroids, boxes):
                self._objects[self.next_id] = (c, b)
                self.next_id += 1
            return {oid: v[1] for oid, v in self._objects.items()}

        old_ids       = list(self._objects.keys())
        old_centroids = [self._objects[i][0] for i in old_ids]

        # Build a sorted list of (distance, old_id, new_idx) for greedy matching
        pairs = sorted(
            (np.hypot(oc[0] - nc[0], oc[1] - nc[1]), oi, ni)
            for oi, oc in zip(old_ids, old_centroids)
            for ni, nc  in enumerate(new_centroids)
        )

        used_old, used_new, updated = set(), set(), {}
        for dist, oi, ni in pairs:
            if dist > self.max_distance:
                break
            if oi in used_old or ni in used_new:
                continue
            updated[oi] = (new_centroids[ni], boxes[ni])
            used_old.add(oi)
            used_new.add(ni)

        for ni, (nc, b) in enumerate(zip(new_centroids, boxes)):
            if ni not in used_new:
                updated[self.next_id] = (nc, b)
                self.next_id += 1

        self._objects = updated
        return {oid: v[1] for oid, v in self._objects.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Helmet Detector
# ──────────────────────────────────────────────────────────────────────────────

class HelmetDetector:
    """
    Per-person helmet detection with centroid tracking and multi-frame smoothing.

    Pipeline per frame
    ------------------
    1. Run a dedicated YOLO model to get helmet bounding boxes.
    2. For each tracked person, build the head region (top 25% of person box).
    3. Check whether any helmet box intersects that head region.
    4. Accumulate the binary result in a per-person history deque.
    5. Majority vote over the history → stable 'has_helmet' flag.

    Required model
    --------------
    The model at *model_path* must be trained to detect helmets.
    The standard yolov8n.pt (COCO) does NOT include a helmet class.
    Use a custom model from Roboflow Universe or similar.
    Set *helmet_class* to the class index your model uses for "helmet".
    """

    def __init__(
        self,
        model_path:   str,
        helmet_class: int = 0,
        history_len:  int = HISTORY_LEN,
    ):
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model        = YOLO(model_path)
        self.helmet_class = helmet_class
        self.tracker      = CentroidTracker()
        self.history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=history_len)
        )
        
        print(f"[HelmetDetector] Model initialized from: {model_path}")
        print(f"[HelmetDetector] Helmet class index: {helmet_class}")

    # ── private ───────────────────────────────────────────────────────────────

    def _get_helmet_boxes(self, frame) -> list:
        results = self.model(frame, verbose=False)[0]
        return [
            tuple(map(int, box.xyxy[0]))
            for box in results.boxes
            if int(box.cls[0]) == self.helmet_class
            and float(box.conf[0]) >= MIN_HELMET_CONF
        ]

    def _person_has_helmet(self, person_box: tuple, helmet_boxes: list) -> bool:
        head = head_region(person_box)
        return any(intersection_area(head, hb) > 0 for hb in helmet_boxes)

    # ── public ────────────────────────────────────────────────────────────────

    def detect(self, frame, persons: list) -> tuple:
        """
        Run helmet detection for one frame.

        Args:
            frame   : BGR numpy array.
            persons : list of (x1, y1, x2, y2) person bounding boxes from YOLO.

        Returns:
            results       : list of dicts —
                            {'id': int, 'box': (x1,y1,x2,y2), 'has_helmet': bool}
            any_violation : True if at least one person has no helmet.
        """
        helmet_boxes = self._get_helmet_boxes(frame)
        tracked      = self.tracker.update(persons)   # {id -> box}
        active_ids   = set(tracked.keys())

        results = []
        for pid, pbox in tracked.items():
            raw      = self._person_has_helmet(pbox, helmet_boxes)
            self.history[pid].append(raw)
            smoothed = sum(self.history[pid]) > len(self.history[pid]) / 2
            results.append({'id': pid, 'box': pbox, 'has_helmet': smoothed})

        # Retire history for persons no longer in frame
        for stale in [k for k in self.history if k not in active_ids]:
            del self.history[stale]

        any_violation = any(not r['has_helmet'] for r in results)
        return results, any_violation


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helper
# ──────────────────────────────────────────────────────────────────────────────

def draw_helmet_results(frame, results: list):
    """
    Draw per-person bounding boxes with 'Helmet' (green) or 'No Helmet' (red).

    Args:
        frame   : BGR numpy array (modified in-place).
        results : list returned by HelmetDetector.detect().
    """
    for r in results:
        x1, y1, x2, y2 = r['box']
        has_helmet = r['has_helmet']
        
        # Green for helmet, Red for no helmet
        color = (0, 255, 0) if has_helmet else (0, 0, 255)
        label = "Helmet" if has_helmet else "No Helmet"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background for better visibility
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame
