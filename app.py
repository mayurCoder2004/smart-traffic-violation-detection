"""
Smart Traffic Violation Detection System
=========================================
Team: 3 Developers
- Developer 1: Helmet Detection + Number Plate Detection + OCR
- Developer 2: Triple Riding Detection
- Developer 3: Overspeed Detection
"""

import os
import re
import time
import math
import threading

import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont

from flask import Flask, Response, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from helmet_detector import HelmetDetector, draw_helmet_results

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Upload Configuration
# ---------------------------------------------------------------------------
UPLOAD_FOLDER      = "uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Current video source: 0 = webcam, string = uploaded file path
_video_source      = 0
_uploaded_filename = None          # original filename shown in the UI


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------------------------
model = YOLO("yolov8n.pt")

# Helmet detection model (custom YOLOv8 trained for helmet detection)
HELMET_MODEL_PATH = "helmet_model.pt"

try:
    _helmet_detector = HelmetDetector(HELMET_MODEL_PATH, helmet_class=0)
    print(f"[helmet] ✓ Model loaded successfully from {HELMET_MODEL_PATH}")
except FileNotFoundError:
    print(f"[helmet] ✗ Model file not found at {HELMET_MODEL_PATH}; helmet detection disabled.")
    _helmet_detector = None
except Exception as exc:
    print(f"[helmet] ✗ Model failed to load ({exc}); helmet detection disabled.")
    _helmet_detector = None

# License-plate detection model
PLATE_MODEL_PATH = "license_plate_detector.pt"

try:
    _plate_model = YOLO(PLATE_MODEL_PATH)
except Exception as exc:
    print(f"[plate] model not loaded ({exc}); plate detection disabled.")
    _plate_model = None

# EasyOCR reader — initialised once; downloads models on first run.
try:
    _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
except Exception as exc:
    print(f"[plate] EasyOCR not loaded ({exc}); OCR disabled.")
    _ocr_reader = None

# ---------------------------------------------------------------------------
# YOLO Class IDs (COCO dataset)
# ---------------------------------------------------------------------------
CLASS_PERSON     = 0
CLASS_CAR        = 2
CLASS_MOTORCYCLE = 3

# Classes passed to YOLO — person, car, motorcycle
YOLO_CLASSES = [CLASS_PERSON, CLASS_CAR, CLASS_MOTORCYCLE]

# Triple-riding threshold
TRIPLE_THRESHOLD = 3

# Colors (BGR)
COLOR_PERSON      = (255, 200,   0)   # gold
COLOR_MOTO_NORMAL = (  0, 255, 100)   # green
COLOR_MOTO_ALERT  = (  0,   0, 255)   # red

# Processing constants
YOLO_IMGSZ          = 256
YOLO_EVERY_N        = 8
PROCESS_WIDTH       = 426
JPEG_QUALITY        = 50
TARGET_FPS          = 30
DETECTION_INTERVAL  = 0.25

# Minimum fraction of a person's area that must overlap a motorcycle box
# for that person to be considered a rider.
_RIDING_OVERLAP_THRESH = 0.15

# Plate detection throttle: OCR runs at most once every N frames.
_PLATE_REFRESH_INTERVAL = 10

# Module-level state for plate caching and frame counting.
_frame_count       = 0
_cached_plate_text = None
_cached_plate_box  = None


def is_riding(person_box: tuple, motorcycle_boxes: list) -> bool:
    """
    Return True if person_box overlaps a motorcycle by at least
    _RIDING_OVERLAP_THRESH of the person's own area.
    """
    px1, py1, px2, py2 = person_box
    person_area = max(1, (px2 - px1) * (py2 - py1))

    for mx1, my1, mx2, my2 in motorcycle_boxes:
        ix1 = max(px1, mx1)
        iy1 = max(py1, my1)
        ix2 = min(px2, mx2)
        iy2 = min(py2, my2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter = (ix2 - ix1) * (iy2 - iy1)
        if inter / person_area >= _RIDING_OVERLAP_THRESH:
            return True
    return False

# ---------------------------------------------------------------------------
# Overspeed Constants  (Dev 3)
# ---------------------------------------------------------------------------
OVERSPEED_LINE1_Y      = 160    # upper trip line y-coordinate
OVERSPEED_LINE2_Y      = 420    # lower trip line y-coordinate
OVERSPEED_REAL_DIST    = 10.0   # real-world distance between lines (metres)
OVERSPEED_LIMIT_KMPH   = 40.0   # speed threshold in km/h
OVERSPEED_MISSING_MAX  = 30     # frames before state auto-resets
OVERSPEED_MAX_JUMP     = 220    # max centroid pixel shift still considered same vehicle


class _VehicleTrack:
    """Per-vehicle timing state — one instance per tracked vehicle."""

    def __init__(self, tid, cx, cy):
        self.id            = tid
        self.last_cx       = cx
        self.last_cy       = cy
        self.missing       = 0
        self.start_time    = None
        self.crossed_line1 = False
        self.crossed_line2 = False
        self.speed_kmph    = None
        self.overspeeding  = False


class _MultiVehicleTracker:
    """
    Centroid-based multi-vehicle tracker.
    Each detected vehicle gets its own _VehicleTrack with an independent timer.
    """

    def __init__(self):
        self.tracks   = {}
        self._next_id = 0

    def reset(self):
        self.tracks   = {}
        self._next_id = 0

    def update(self, boxes):
        """
        Match boxes to existing tracks by nearest centroid.
        Returns [(box, track), ...] for every currently active detection.
        """
        centroids    = [((x1+x2)//2, (y1+y2)//2) for x1,y1,x2,y2 in boxes]
        box_to_track = {}
        used_boxes   = set()

        for tid in list(self.tracks):
            track = self.tracks.get(tid)
            if track is None:
                continue

            best_i, best_d = -1, float("inf")
            for i, (cx, cy) in enumerate(centroids):
                if i in used_boxes:
                    continue
                d = math.hypot(cx - track.last_cx, cy - track.last_cy)
                if d < best_d:
                    best_d, best_i = d, i

            if best_i >= 0 and best_d <= OVERSPEED_MAX_JUMP:
                track.last_cx = centroids[best_i][0]
                track.last_cy = centroids[best_i][1]
                track.missing = 0
                used_boxes.add(best_i)
                box_to_track[best_i] = track
            else:
                track.missing += 1
                if track.missing > OVERSPEED_MISSING_MAX:
                    del self.tracks[tid]

        for i, (cx, cy) in enumerate(centroids):
            if i not in used_boxes:
                tid   = self._next_id
                self._next_id += 1
                track = _VehicleTrack(tid, cx, cy)
                self.tracks[tid] = track
                box_to_track[i]  = track

        return [(boxes[i], track) for i, track in sorted(box_to_track.items())]


_tracker = _MultiVehicleTracker()

# ---------------------------------------------------------------------------
# Shared Frame Buffer (background capture thread writes, Flask thread reads)
# ---------------------------------------------------------------------------
_latest_frame  = None
_frame_lock    = threading.Lock()
_worker_source = None           # sentinel: worker exits when this changes

# ---------------------------------------------------------------------------
# Helmet Module  (Dev 1)
# ---------------------------------------------------------------------------

def detect_helmet(frame, persons):
    """
    Detect whether each person is wearing a helmet.

    Returns:
        results       (list[dict]): Per-person dicts with 'id', 'box', 'has_helmet'.
        any_violation (bool):       True if at least one person has no helmet.
    """
    if _helmet_detector is None:
        return [], False
    return _helmet_detector.detect(frame, persons)


# ---------------------------------------------------------------------------
# Plate Module  (Dev 1)
# ---------------------------------------------------------------------------

def detect_number_plate(frame):
    """
    Detect and read the number plate from the frame.

    Returns:
        plate_text (str | None): OCR-extracted plate text, or None if not found.
        plate_box  (tuple | None): Bounding box (x1,y1,x2,y2), or None.
    """
    if _plate_model is None:
        return None, None

    results = _plate_model(frame, verbose=False)[0]
    if not results.boxes:
        return None, None

    best = max(results.boxes, key=lambda b: float(b.conf[0]))
    if float(best.conf[0]) < 0.4:
        return None, None

    x1, y1, x2, y2 = map(int, best.xyxy[0])
    plate_box = (x1, y1, x2, y2)

    if _ocr_reader is None:
        return None, plate_box

    plate_crop = frame[y1:y2, x1:x2]
    if plate_crop.size == 0:
        return None, plate_box

    ocr_results = _ocr_reader.readtext(plate_crop)
    if not ocr_results:
        return None, plate_box

    best_ocr   = max(ocr_results, key=lambda r: r[2])
    plate_text = re.sub(r"[^A-Z0-9]", "", best_ocr[1].upper())

    return plate_text, plate_box


# ---------------------------------------------------------------------------
# Triple Riding Module  (Dev 2)
# ---------------------------------------------------------------------------

def is_person_on_motorcycle(person_box, moto_box, vertical_margin=0.25):
    """Return True if a person is likely riding the given motorcycle."""
    px1, py1, px2, py2 = person_box
    mx1, my1, mx2, my2 = moto_box

    p_w = px2 - px1
    p_h = py2 - py1
    m_w = mx2 - mx1

    if p_w >= p_h:          # real persons are taller than wide
        return False
    if p_w > m_w * 0.80:   # rider shouldn't be wider than the motorcycle
        return False
    if not (px1 < mx2 and px2 > mx1):  # horizontal overlap required
        return False

    moto_height = my2 - my1
    upper_limit = my1 - vertical_margin * moto_height
    return py1 <= my2 and py2 >= upper_limit


def detect_triple_riding(frame, persons, motorcycles):
    """
    Detect triple riding (3+ persons on a single motorcycle).

    Includes a fallback: if YOLO misses the motorcycle but finds 3+ persons,
    a synthetic motorcycle box is derived from the person cluster.

    Returns:
        bool: True if triple riding is detected on any motorcycle.
    """
    # Synthesize a motorcycle box from person cluster when none is detected
    effective_motos = list(motorcycles)
    if not effective_motos and len(persons) >= TRIPLE_THRESHOLD:
        sx1 = min(p[0] for p in persons)
        sy1 = min(p[1] for p in persons)
        sx2 = max(p[2] for p in persons)
        sy2 = max(p[3] for p in persons)
        effective_motos = [(sx1, sy1, sx2, sy2)]

    any_triple = False

    for (mx1, my1, mx2, my2) in effective_motos:
        rider_count = sum(
            1 for p in persons
            if is_person_on_motorcycle(p, (mx1, my1, mx2, my2))
        )
        triple = rider_count >= TRIPLE_THRESHOLD

        if triple:
            any_triple = True
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), COLOR_MOTO_ALERT, 3)
            cv2.putText(frame, "Triple Riding Detected!",
                        (mx1, my1 - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_MOTO_ALERT, 2)
            cv2.putText(frame, f"Riders: {rider_count}  |  Fine: Rs.1000",
                        (mx1, my1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_MOTO_ALERT, 2)
        else:
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), COLOR_MOTO_NORMAL, 2)
            cv2.putText(frame, f"Motorcycle  Riders: {rider_count}",
                        (mx1, my1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_MOTO_NORMAL, 2)

    if any_triple:
        cv2.rectangle(frame, (0, 0), (360, 60), (0, 0, 200), -1)
        cv2.putText(frame, "TRIPLE RIDING DETECTED", (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(frame, "Fine: Rs.1000", (8, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

    return any_triple


# ---------------------------------------------------------------------------
# Overspeed Module  (Dev 3)
# ---------------------------------------------------------------------------

def detect_overspeed(frame, vehicles):
    """
    Detect ALL visible vehicles exceeding OVERSPEED_LIMIT_KMPH using two trip lines.

    Side-effects: draws trip lines and per-vehicle speed labels onto frame.

    Returns:
        bool: True if ANY tracked vehicle exceeded OVERSPEED_LIMIT_KMPH.
    """
    w = frame.shape[1]

    cv2.line(frame, (0, OVERSPEED_LINE1_Y), (w, OVERSPEED_LINE1_Y), (0, 200, 255), 2)
    cv2.putText(frame, "LINE 1", (10, OVERSPEED_LINE1_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
    cv2.line(frame, (0, OVERSPEED_LINE2_Y), (w, OVERSPEED_LINE2_Y), (255, 200, 0), 2)
    cv2.putText(frame, "LINE 2", (10, OVERSPEED_LINE2_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    matched       = _tracker.update(vehicles)
    any_overspeed = False

    for box, track in matched:
        x1, y1, x2, y2 = box
        cx, cy          = track.last_cx, track.last_cy

        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        if not track.crossed_line2:
            if not track.crossed_line1 and cy >= OVERSPEED_LINE1_Y:
                track.start_time    = time.time()
                track.crossed_line1 = True
                print(f"[Track {track.id}] Started timing.")

            elif track.crossed_line1 and cy >= OVERSPEED_LINE2_Y:
                elapsed              = time.time() - track.start_time
                track.speed_kmph     = round((OVERSPEED_REAL_DIST / elapsed) * 3.6, 2)
                track.overspeeding   = track.speed_kmph > OVERSPEED_LIMIT_KMPH
                track.crossed_line2  = True
                print(f"[Track {track.id}] {track.speed_kmph} km/h — "
                      f"{'OVERSPEEDING' if track.overspeeding else 'OK'}")

        if track.overspeeding:
            any_overspeed = True

        if track.speed_kmph is not None:
            color = (0, 0, 255) if track.overspeeding else (0, 220, 0)
            cv2.putText(frame, f"{track.speed_kmph:.1f} km/h",
                        (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif track.crossed_line1:
            elapsed = time.time() - track.start_time
            cv2.putText(frame, f"T{track.id}: {elapsed:.1f}s",
                        (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2)

    return any_overspeed


# ---------------------------------------------------------------------------
# YOLO Detection Helper
# ---------------------------------------------------------------------------

def run_yolo_detection(frame):
    """
    Run YOLOv8 on a frame and extract persons, motorcycles, and cars.

    Returns:
        persons     (list[tuple]): [(x1,y1,x2,y2), ...]
        motorcycles (list[tuple]): [(x1,y1,x2,y2), ...]
        cars        (list[tuple]): [(x1,y1,x2,y2), ...]
    """
    results = model(
        frame,
        verbose=False,
        imgsz=YOLO_IMGSZ,
        classes=YOLO_CLASSES,
        conf=0.4,
    )[0]

    persons, motorcycles, cars = [], [], []

    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.25:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        coords = (x1, y1, x2, y2)

        if cls == CLASS_PERSON:
            persons.append(coords)
        elif cls == CLASS_MOTORCYCLE:
            motorcycles.append(coords)
        elif cls == CLASS_CAR:
            cars.append(coords)

    return persons, motorcycles, cars


# ---------------------------------------------------------------------------
# Drawing Utilities
# ---------------------------------------------------------------------------

def draw_boxes(frame, boxes, color, label=""):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _get_pil_font(size: int) -> ImageFont.FreeTypeFont:
    """Return a TTF font that supports Unicode (₹, ❌). Falls back to default."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()

_overlay_font = None


def draw_violation_overlay(frame, helmet_violation, triple_violation,
                           overspeed_violation, plate_text):
    """
    Draw a challan info box using PIL so Unicode characters (❌, ₹) render correctly.
    Modifies frame in-place.
    """
    global _overlay_font

    any_violation = helmet_violation or triple_violation or overspeed_violation
    if not any_violation:
        return

    lines = []
    if helmet_violation:
        lines.append(("No Helmet ❌", (255, 80, 80)))
    if triple_violation:
        lines.append(("Triple Riding !!!", (255, 120, 0)))
    if overspeed_violation:
        lines.append(("Overspeed >>>", (255, 165, 0)))

    plate_display = plate_text if plate_text else "Detecting..."
    lines.append((f"Plate: {plate_display}", (255, 220, 0)))

    fine_amt = "₹1000" if (triple_violation or overspeed_violation) else "₹500"
    lines.append((f"Fine: {fine_amt}", (255, 220, 0)))

    if _overlay_font is None:
        _overlay_font = _get_pil_font(22)

    padding, line_h = 10, 30
    box_w = 280
    box_h = len(lines) * line_h + padding * 2

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)

    draw.rectangle([8, 8, 8 + box_w, 8 + box_h], fill=(15, 15, 15))

    for i, (text, colour) in enumerate(lines):
        draw.text(
            (8 + padding, 8 + padding + i * line_h),
            text, fill=colour, font=_overlay_font,
        )

    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Main Frame-Processing Pipeline  (all three detections in one pass)
# ---------------------------------------------------------------------------

def resize_for_processing(frame, target_width=PROCESS_WIDTH):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def process_frame(frame):
    """
    Full detection pipeline:
      Dev 1 — Helmet detection + Number plate OCR
      Dev 2 — Triple riding detection
      Dev 3 — Overspeed detection

    Modifies frame in-place with annotations and returns it.
    """
    global _frame_count, _cached_plate_text, _cached_plate_box
    _frame_count += 1

    # --- YOLO Detection (persons, motorcycles, cars) ---
    persons, motorcycles, cars = run_yolo_detection(frame)
    vehicles = motorcycles + cars

    # --- Filter: only persons overlapping a motorcycle are checked for helmets ---
    riders     = [p for p in persons if is_riding(p, motorcycles)]
    non_riders = [p for p in persons if not is_riding(p, motorcycles)]

    # --- Dev 1: Helmet detection (riders only) ---
    helmet_results, helmet_violation = detect_helmet(frame, riders)

    # --- Dev 2: Triple riding detection ---
    triple_violation = detect_triple_riding(frame, persons, motorcycles)

    # --- Dev 3: Overspeed detection ---
    overspeed_violation = detect_overspeed(frame, vehicles)

    # --- Unified violation flag ---
    violation = helmet_violation or triple_violation or overspeed_violation

    # --- Dev 1: Plate detection (only when there is a violation, throttled) ---
    if violation:
        if _frame_count % _PLATE_REFRESH_INTERVAL == 0:
            _cached_plate_text, _cached_plate_box = detect_number_plate(frame)
        plate_text = _cached_plate_text
        plate_box  = _cached_plate_box
    else:
        _cached_plate_text = None
        _cached_plate_box  = None
        plate_text = None
        plate_box  = None

    # --- Draw bounding boxes ---
    draw_boxes(frame, non_riders, color=COLOR_PERSON, label="Person")

    if helmet_results:
        draw_helmet_results(frame, helmet_results)
    else:
        draw_boxes(frame, riders, color=COLOR_PERSON, label="Rider")

    # Motorcycles are drawn by detect_triple_riding; draw cars separately.
    draw_boxes(frame, cars, color=(0, 180, 255), label="Car")

    if plate_box:
        draw_boxes(frame, [plate_box], color=(0, 255, 255), label="Plate")
        if plate_text:
            x1, y1, _, _ = plate_box
            cv2.putText(frame, f"Plate: {plate_text}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # --- Violation overlay (challan box) ---
    draw_violation_overlay(frame, helmet_violation, triple_violation,
                           overspeed_violation, plate_text)

    return frame


# ---------------------------------------------------------------------------
# Background Capture Worker  (writes to shared _latest_frame buffer)
# ---------------------------------------------------------------------------

def _capture_worker(source):
    """
    Runs in a daemon thread. Captures frames, runs the full pipeline,
    and stores the latest result in _latest_frame.
    Exits automatically when _worker_source changes (source switches).
    """
    global _latest_frame

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return

    consecutive_errors = 0
    try:
        while _worker_source == source:
            ret, frame = cap.read()

            if not ret:
                if isinstance(source, str):     # video file ended → loop
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    _tracker.reset()
                    consecutive_errors = 0
                    continue
                # Webcam transient error — retry before giving up
                consecutive_errors += 1
                if consecutive_errors > 60:     # ~2 s of continuous failure
                    break
                time.sleep(0.033)
                continue

            consecutive_errors = 0

            frame = cv2.resize(frame, (854, 480))
            frame = process_frame(frame)

            with _frame_lock:
                _latest_frame = frame
    finally:
        cap.release()


def _start_capture(source):
    """Switch to a new source: update the sentinel and spawn a fresh daemon thread."""
    global _worker_source, _latest_frame

    _worker_source = source
    with _frame_lock:
        _latest_frame = None

    t = threading.Thread(target=_capture_worker, args=(source,), daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Async helpers  (Dev 2 — used by streamlit_app.py for non-Flask UI)
# ---------------------------------------------------------------------------

class LatestFrameCapture:
    """Wraps a cv2.VideoCapture and keeps the most recent frame in a lock."""

    def __init__(self, cap):
        self.cap     = cap
        self.frame   = None
        self.ok      = cap.isOpened()
        self.lock    = threading.Lock()
        self.stopped = False
        self.thread  = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                self.ok = False
                time.sleep(0.01)
                continue
            with self.lock:
                self.ok    = True
                self.frame = frame

    def isOpened(self):
        return self.cap.isOpened()

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
    """Runs YOLO inference in a background thread to avoid blocking the display loop."""

    def __init__(self):
        self.input_frame  = None
        self.persons      = []
        self.motorcycles  = []
        self.lock         = threading.Lock()
        self.stopped      = False
        self.thread       = threading.Thread(target=self._worker, daemon=True)
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

            persons, motorcycles, _ = run_yolo_detection(frame)
            with self.lock:
                self.persons     = persons
                self.motorcycles = motorcycles

            time.sleep(DETECTION_INTERVAL)

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1)


# ---------------------------------------------------------------------------
# Flask Streaming
# ---------------------------------------------------------------------------

def generate_frames():
    """
    Read the latest processed frame from the shared buffer and yield it as
    MJPEG. Decoupled from YOLO — runs at up to 30 fps regardless of inference speed.
    """
    while True:
        with _frame_lock:
            frame = _latest_frame

        if frame is None:
            time.sleep(0.02)
            continue

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if ret:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

        time.sleep(0.033)   # cap stream at ~30 fps


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    response = Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control":     "no-cache, no-store, must-revalidate",
            "Pragma":            "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
    return response


@app.route("/upload", methods=["POST"])
def upload_video():
    """Accept a video file, save it, and switch the stream to that file."""
    global _video_source, _uploaded_filename

    if "video" not in request.files:
        return jsonify({"success": False, "error": "No file in request."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS).upper()}"
        }), 400

    original_name = secure_filename(file.filename)
    ext           = original_name.rsplit(".", 1)[1].lower()
    save_path     = os.path.join(UPLOAD_FOLDER, f"current_video.{ext}")

    file.save(save_path)

    _video_source      = save_path
    _uploaded_filename = original_name
    _tracker.reset()
    _start_capture(save_path)

    return jsonify({"success": True, "filename": original_name})


@app.route("/use_webcam", methods=["POST"])
def use_webcam():
    """Switch back to the live webcam feed."""
    global _video_source, _uploaded_filename

    _video_source      = 0
    _uploaded_filename = None
    _tracker.reset()
    _start_capture(0)

    return jsonify({"success": True})


@app.route("/source_status")
def source_status():
    """Return the current source label so the UI can show it."""
    if _video_source == 0:
        return jsonify({"source": "webcam", "label": "Live Webcam"})
    return jsonify({"source": "file", "label": _uploaded_filename or "Uploaded Video"})


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _start_capture(0)                               # start webcam worker on launch
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
