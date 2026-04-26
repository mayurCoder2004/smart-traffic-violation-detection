"""
Triple Riding Detection — Standalone Script
============================================
Developer 2 Module

Detects when 3 or more persons are riding a single motorcycle using
YOLOv8n (COCO pretrained) and OpenCV.

Usage:
    python triple_riding.py              # webcam
    python triple_riding.py video.mp4   # video file
"""

import sys
import time
import threading
import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.40          # minimum detection confidence
TRIPLE_RIDING_THRESHOLD = 3    # persons per motorcycle to trigger alert
YOLO_IMGSZ = 256               # smaller input keeps webcam inference responsive
YOLO_EVERY_N = 8               # reuse detections between YOLO passes
PROCESS_WIDTH = 426            # downscale camera frames before inference/display
DETECTION_INTERVAL = 0.25

# COCO class IDs
CLASS_PERSON     = 0
CLASS_MOTORCYCLE = 3
YOLO_CLASSES = [CLASS_PERSON, CLASS_MOTORCYCLE]

# Colors (BGR)
COLOR_PERSON        = (255, 200, 0)    # gold
COLOR_MOTO_NORMAL   = (0, 255, 100)   # green
COLOR_MOTO_ALERT    = (0, 0, 255)     # red  — triple riding
COLOR_TEXT          = (255, 255, 255) # white


# ---------------------------------------------------------------------------
# Association Logic
# ---------------------------------------------------------------------------

def is_person_on_motorcycle(person_box, moto_box, vertical_margin=0.25):
    """
    Return True if a person is likely riding the given motorcycle.

    Criteria
    --------
    1. Aspect ratio  : person box must be taller than wide (vehicles are wider than tall).
    2. Width ratio   : person must not be significantly wider than the motorcycle.
    3. Horizontal overlap : person x-range overlaps motorcycle x-range.
    4. Vertical position  : person is above or overlapping the motorcycle, not below it.
    """
    px1, py1, px2, py2 = person_box
    mx1, my1, mx2, my2 = moto_box

    p_w = px2 - px1
    p_h = py2 - py1
    m_w = mx2 - mx1

    # --- 1. Aspect ratio guard: real persons are taller than wide ---
    # Vehicles (cars, bikes) have width > height, so this filters them out.
    if p_w >= p_h:
        return False

    # --- 2. Width ratio guard: a rider shouldn't be wider than the motorcycle ---
    # Allows up to 80% of motorcycle width to handle side-by-side riders.
    if p_w > m_w * 0.80:
        return False

    # --- 3. Horizontal overlap check ---
    if not (px1 < mx2 and px2 > mx1):
        return False

    # --- 4. Vertical position check ---
    moto_height = my2 - my1
    upper_limit = my1 - vertical_margin * moto_height
    v_valid = py1 <= my2 and py2 >= upper_limit
    return v_valid


# ---------------------------------------------------------------------------
# Per-Frame Detection
# ---------------------------------------------------------------------------

def resize_frame(frame, target_width=PROCESS_WIDTH):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def process_frame(frame, model, state=None):
    """
    Run YOLO, evaluate triple-riding for each motorcycle, draw results.

    Returns the annotated frame.
    """
    if state is None:
        state = {}

    state["frame_count"] = state.get("frame_count", 0) + 1
    should_detect = (
        state["frame_count"] == 1
        or state["frame_count"] % YOLO_EVERY_N == 0
        or "persons" not in state
    )

    if should_detect:
        results = model(
            frame,
            verbose=False,
            imgsz=YOLO_IMGSZ,
            classes=YOLO_CLASSES,
            conf=CONF_THRESHOLD,
        )[0]

        persons     = []
        motorcycles = []

        # --- Parse YOLO detections ---
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == CLASS_PERSON:
                persons.append((x1, y1, x2, y2))
            elif cls == CLASS_MOTORCYCLE:
                motorcycles.append((x1, y1, x2, y2))

        state["persons"] = persons
        state["motorcycles"] = motorcycles
    else:
        persons = state.get("persons", [])
        motorcycles = state.get("motorcycles", [])

    # --- Draw all person boxes ---
    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)
        cv2.putText(frame, "Person", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PERSON, 2)

    # --- Evaluate each motorcycle for triple riding ---
    any_triple = False

    for (mx1, my1, mx2, my2) in motorcycles:
        # Count persons associated with this motorcycle
        rider_count = sum(
            1 for p in persons if is_person_on_motorcycle(p, (mx1, my1, mx2, my2))
        )

        triple = rider_count >= TRIPLE_RIDING_THRESHOLD

        if triple:
            any_triple = True
            color = COLOR_MOTO_ALERT
            # Draw motorcycle box in red
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), color, 3)
            # Alert label just above the box
            cv2.putText(frame, "Triple Riding Detected!", (mx1, my1 - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(frame, f"Riders: {rider_count}  |  Fine: Rs.1000",
                        (mx1, my1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        else:
            color = COLOR_MOTO_NORMAL
            # Draw motorcycle box in green
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), color, 2)
            label = f"Motorcycle  Riders: {rider_count}"
            cv2.putText(frame, label, (mx1, my1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # --- Top-left summary banner when a violation is active ---
    if any_triple:
        cv2.rectangle(frame, (0, 0), (360, 60), (0, 0, 200), -1)  # filled red banner
        cv2.putText(frame, "TRIPLE RIDING DETECTED", (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, COLOR_TEXT, 2)
        cv2.putText(frame, "Fine: Rs.1000", (8, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

    return frame


def draw_cached_detection(frame, persons, motorcycles):
    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)
        cv2.putText(frame, "Person", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PERSON, 2)

    any_triple = False
    for (mx1, my1, mx2, my2) in motorcycles:
        rider_count = sum(
            1 for p in persons if is_person_on_motorcycle(p, (mx1, my1, mx2, my2))
        )
        triple = rider_count >= TRIPLE_RIDING_THRESHOLD
        color = COLOR_MOTO_ALERT if triple else COLOR_MOTO_NORMAL
        if triple:
            any_triple = True

        cv2.rectangle(frame, (mx1, my1), (mx2, my2), color, 3 if triple else 2)
        label = (
            f"Riders: {rider_count}  |  Fine: Rs.1000"
            if triple
            else f"Motorcycle  Riders: {rider_count}"
        )
        cv2.putText(frame, label, (mx1, my1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if any_triple:
        cv2.rectangle(frame, (0, 0), (360, 60), (0, 0, 200), -1)
        cv2.putText(frame, "TRIPLE RIDING DETECTED", (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, COLOR_TEXT, 2)
        cv2.putText(frame, "Fine: Rs.1000", (8, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

    return frame


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

class LatestFrameCapture:
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ok = cap.isOpened()
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                self.ok = False
                time.sleep(0.01)
                continue
            with self.lock:
                self.ok = True
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()


class AsyncYoloDetector:
    def __init__(self, model):
        self.model = model
        self.input_frame = None
        self.persons = []
        self.motorcycles = []
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.lock:
            self.input_frame = frame.copy()

    def get_result(self):
        with self.lock:
            return list(self.persons), list(self.motorcycles)

    def _worker(self):
        while not self.stopped:
            with self.lock:
                frame = None if self.input_frame is None else self.input_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            results = self.model(
                frame,
                verbose=False,
                imgsz=YOLO_IMGSZ,
                classes=YOLO_CLASSES,
                conf=CONF_THRESHOLD,
            )[0]

            persons, motorcycles = [], []
            for box in results.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls == CLASS_PERSON:
                    persons.append((x1, y1, x2, y2))
                elif cls == CLASS_MOTORCYCLE:
                    motorcycles.append((x1, y1, x2, y2))

            with self.lock:
                self.persons = persons
                self.motorcycles = motorcycles

            time.sleep(DETECTION_INTERVAL)

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0  # 0 = default webcam

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"[INFO] Opening video source: {source}")
    raw_cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if source == 0 else cv2.VideoCapture(source)

    if not raw_cap.isOpened():
        print("[ERROR] Cannot open video source.")
        sys.exit(1)

    raw_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    raw_cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_WIDTH)
    raw_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap = LatestFrameCapture(raw_cap) if source == 0 else raw_cap

    print("[INFO] Press 'q' to quit.")

    state = {}
    detector = AsyncYoloDetector(model) if source == 0 else None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if source == 0:
                    time.sleep(0.01)
                    continue
                print("[INFO] End of stream.")
                break

            frame = resize_frame(frame)
            if detector is not None:
                detector.submit(frame)
                persons, motorcycles = detector.get_result()
                frame = draw_cached_detection(frame, persons, motorcycles)
            else:
                frame = process_frame(frame, model, state)
            cv2.imshow("Triple Riding Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if detector is not None:
            detector.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
