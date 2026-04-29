"""
Smart Traffic Signal System
============================
Detects vehicles via YOLOv8, assigns them to two predefined polygon zones
(Lane A = top road region, Lane B = right road region), and dynamically
controls signal timing based on vehicle density.

Signal rules
------------
* Only ONE lane is GREEN at a time.
* First cycle → busier lane gets GREEN first.
* After each timer → switch to the other lane.
* Green duration: 0-10 vehicles→10s  |  11-20→20s  |  21-30→30s
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO

# ── COCO vehicle class IDs ────────────────────────────────────────────────────
VEHICLE_CLASSES = {2: "car", 3: "bike", 5: "bus", 7: "truck"}

# ── Hardcoded polygon zones for 854×480 frames ────────────────────────────────
# Lane A — Top road region (vehicles entering from the top of the frame)
LANE_A_POLY = np.array([
    [ 80,   4],
    [774,   4],
    [718, 218],
    [136, 218],
], np.int32)

# Lane B — Right road region (vehicles entering from the right of the frame)
LANE_B_POLY = np.array([
    [592, 148],
    [850, 105],
    [850, 476],
    [568, 476],
], np.int32)

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
_CLR_LANE_A = (  0, 220, 180)   # teal
_CLR_LANE_B = (255, 155,   0)   # amber
_CLR_OTHER  = (160, 160, 160)   # grey  (vehicles outside both zones)
_CLR_GREEN  = (  0, 215,   0)
_CLR_RED    = (  0,   0, 220)
_CLR_PANEL  = ( 16,  16,  20)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_signal_time(vehicle_count: int) -> int:
    """Return green-signal duration (seconds) for a given vehicle count.

    Range        → Duration
    0  –  5      → 10 s
    6  – 10      → 20 s
    11 – 15      → 30 s
    16 – 20      → 40 s
    > 20         → 50 s  (max cap)
    """
    if vehicle_count <= 5:
        return 10
    if vehicle_count <= 10:
        return 20
    if vehicle_count <= 15:
        return 30
    if vehicle_count <= 20:
        return 40
    return 50


def _in_zone(cx: int, cy: int, poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


# ── Main class ────────────────────────────────────────────────────────────────

class TrafficSignalSystem:
    """
    Self-contained smart traffic signal controller.

    Usage:
        system = TrafficSignalSystem()
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (854, 480))
            frame = system.process_frame(frame)
            ...
    Public attribute ``state`` holds the latest JSON-serialisable status dict.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

        self._green_lane = None
        self._green_end:  float      = 0.0
        self._green_dur:  int        = 0

        self.state: dict = {
            "lane_a_count": 0,
            "lane_b_count": 0,
            "green_lane":   None,
            "remaining":    0,
            "green_dur":    0,
        }

    # ── YOLO detection ────────────────────────────────────────────────────────

    def _detect(self, frame) -> list[tuple]:
        """Return list of (x1,y1,x2,y2,class_label) for each vehicle."""
        results = self.model(
            frame,
            verbose=False,
            conf=0.35,
            classes=list(VEHICLE_CLASSES.keys()),
        )[0]
        vehicles = []
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.35 or cls not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicles.append((x1, y1, x2, y2, VEHICLE_CLASSES[cls]))
        return vehicles

    # ── Zone assignment ───────────────────────────────────────────────────────

    def _assign(self, vehicles):
        """Split detected vehicles into Lane A, Lane B, and unassigned."""
        lane_a, lane_b, other = [], [], []
        for v in vehicles:
            x1, y1, x2, y2, cls = v
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if _in_zone(cx, cy, LANE_A_POLY):
                lane_a.append(v)
            elif _in_zone(cx, cy, LANE_B_POLY):
                lane_b.append(v)
            else:
                other.append(v)
        return lane_a, lane_b, other

    # ── Signal state machine ──────────────────────────────────────────────────

    def _tick(self, count_a: int, count_b: int) -> tuple[str, int]:
        """
        Advance the signal clock and return (green_lane, remaining_seconds).
        Switches to the other lane when the timer expires.
        """
        now = time.time()

        if now < self._green_end:
            return self._green_lane, int(self._green_end - now)

        # Timer expired (or first run) → choose next green lane
        if self._green_lane is None:
            # First cycle: give green to the busier lane
            self._green_lane = 'A' if count_a >= count_b else 'B'
        else:
            # Subsequent cycles: alternate lanes
            self._green_lane = 'B' if self._green_lane == 'A' else 'A'

        count            = count_a if self._green_lane == 'A' else count_b
        self._green_dur  = get_signal_time(count)
        self._green_end  = now + self._green_dur

        return self._green_lane, self._green_dur

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _draw(self, frame, lane_a, lane_b, other, green_lane: str, remaining: int):
        h, w = frame.shape[:2]

        # ── Filled semi-transparent zone overlays ─────────────────────────────
        overlay = frame.copy()
        cv2.fillPoly(overlay, [LANE_A_POLY], (*_CLR_LANE_A[::-1][::-1], ))   # draw on overlay
        cv2.fillPoly(overlay, [LANE_B_POLY], (*_CLR_LANE_B[::-1][::-1], ))
        cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

        # ── Polygon outlines ──────────────────────────────────────────────────
        cv2.polylines(frame, [LANE_A_POLY], True, _CLR_LANE_A, 2)
        cv2.polylines(frame, [LANE_B_POLY], True, _CLR_LANE_B, 2)

        # Zone labels (inside the polygon, top corner)
        cv2.putText(frame, "LANE A",
                    (LANE_A_POLY[0][0] + 6, LANE_A_POLY[0][1] + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, _CLR_LANE_A, 2)
        cv2.putText(frame, "LANE B",
                    (LANE_B_POLY[0][0] + 6, LANE_B_POLY[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, _CLR_LANE_B, 2)

        # ── Vehicle bounding boxes & centre dots ──────────────────────────────
        for grp, clr in ((lane_a, _CLR_LANE_A), (lane_b, _CLR_LANE_B), (other, _CLR_OTHER)):
            for x1, y1, x2, y2, cls in grp:
                cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 4, clr, -1)
                cv2.putText(frame, cls, (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, clr, 2)

        # ── Signal status panel (bottom-left) ─────────────────────────────────
        PW, PH = 300, 175
        px, py = 10, h - PH - 10

        bg = frame.copy()
        cv2.rectangle(bg, (px, py), (px + PW, py + PH), _CLR_PANEL, -1)
        cv2.addWeighted(bg, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (px, py), (px + PW, py + PH), (70, 70, 70), 1)

        # Title
        cv2.putText(frame, "TRAFFIC SIGNAL", (px + 10, py + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Lane A row
        a_clr   = _CLR_GREEN if green_lane == 'A' else _CLR_RED
        a_label = f"LANE A : GO  {remaining}s" if green_lane == 'A' else "LANE A : STOP"
        cv2.circle(frame, (px + 22, py + 58), 13, a_clr, -1)
        cv2.putText(frame, a_label, (px + 44, py + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, a_clr, 2)

        # Lane B row
        b_clr   = _CLR_GREEN if green_lane == 'B' else _CLR_RED
        b_label = f"LANE B : GO  {remaining}s" if green_lane == 'B' else "LANE B : STOP"
        cv2.circle(frame, (px + 22, py + 100), 13, b_clr, -1)
        cv2.putText(frame, b_label, (px + 44, py + 107),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, b_clr, 2)

        # Vehicle counts
        cv2.putText(frame,
                    f"Lane A: {len(lane_a)} vehicles   Lane B: {len(lane_b)} vehicles",
                    (px + 10, py + 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        # Duration rule reminder
        dur_text = "0-5=10s | 6-10=20s | 11-15=30s | 16-20=40s | >20=50s"
        cv2.putText(frame, dur_text, (px + 10, py + 166),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1)

        return frame

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run the full pipeline on one frame.
        Modifies frame in-place with annotations and returns it.
        Updates ``self.state`` with the latest signal info.
        """
        vehicles             = self._detect(frame)
        lane_a, lane_b, ot  = self._assign(vehicles)
        green_lane, rem      = self._tick(len(lane_a), len(lane_b))
        frame                = self._draw(frame, lane_a, lane_b, ot, green_lane, rem)

        self.state = {
            "lane_a_count": len(lane_a),
            "lane_b_count": len(lane_b),
            "green_lane":   green_lane,
            "remaining":    rem,
            "green_dur":    self._green_dur,
        }
        return frame


# ── Standalone runner ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    src_arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    src     = int(src_arg) if src_arg.isdigit() else src_arg

    cap    = cv2.VideoCapture(src)
    system = TrafficSignalSystem()

    print("Smart Traffic Signal System — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(src, str):          # video file — loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame = cv2.resize(frame, (854, 480))
        frame = system.process_frame(frame)

        s = system.state
        print(f"\r[Lane A: {s['lane_a_count']:2d}]  "
              f"[Lane B: {s['lane_b_count']:2d}]  "
              f"GREEN → {s['green_lane']}  "
              f"({s['remaining']:2d}s remaining)   ", end="", flush=True)

        cv2.imshow("Smart Traffic Signal System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print()
