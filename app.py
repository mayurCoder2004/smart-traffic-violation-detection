"""
Smart Traffic Violation Detection System
=========================================
Team: 3 Developers
- Developer 1: Helmet Detection + Number Plate Detection + OCR
- Developer 2: Triple Riding Detection
- Developer 3: Overspeed Detection
"""

import os
import random
import re
import time
import math
import queue
import threading
import hmac
import hashlib
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from helmet_detector import HelmetDetector, draw_helmet_results

load_dotenv()

IST = timezone(timedelta(hours=5, minutes=30))

try:
    import razorpay as _razorpay_lib
    _RAZORPAY_AVAILABLE = True
except ImportError:
    _RAZORPAY_AVAILABLE = False

app = Flask(__name__)

# ── CORS — applied unconditionally so /video, /detections etc always work ────
# Must come BEFORE blueprint registration so Flask-CORS sees all routes.
_CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]
CORS(app, origins=_CORS_ORIGINS, supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"])

# ── Database + Blueprint wiring ──────────────────────────────────────────────
try:
    from backend.config import Config
    from backend.extensions import db
    from backend.models import (
        User as _DBUser,
        ScannerChallan as _DBScannerChallan,
        ScannerChallanItem as _DBScannerChallanItem,
    )
    from backend.routes.users import users_bp
    from backend.routes.violations import violations_bp
    from backend.routes.payments import payments_bp

    app.config.from_object(Config)
    db.init_app(app)

    app.register_blueprint(users_bp)
    app.register_blueprint(violations_bp)
    app.register_blueprint(payments_bp)

    _DB_ENABLED = True
    print("[DB] Backend modules loaded successfully.")
except Exception as _db_err:
    print(f"[DB] Backend modules not loaded ({_db_err}). Running without DB.")
    _DB_ENABLED = False
    _DBUser      = None
    _DBScannerChallan = None
    _DBScannerChallanItem = None

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

# Traffic signal system (loaded once, shared across signal routes)
from traffic_signal import TrafficSignalSystem as _TrafficSignalSystem
_signal_system = _TrafficSignalSystem("yolov8n.pt")

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

# Frame counter — incremented every process_frame call.
_frame_count = 0

# ── Async violation reporter ─────────────────────────────────────────────────
# Violations detected by CV are queued here; a background thread writes to DB.
_violation_q: queue.Queue = queue.Queue(maxsize=200)
_plate_cooldowns: dict = {}          # (plate, vtype) -> monotonic timestamp
VIOLATION_COOLDOWN_SECS = 10         # minimum seconds between same violation/plate
_FINE_MAP = {"helmet": 500, "triple": 1000, "overspeed": 1000}
_VIOLATION_LABELS = {
    "helmet":    "No Helmet",
    "triple":    "Triple Riding",
    "overspeed": "Overspeed",
}
_reporter_thread = None
_reporter_lock = threading.Lock()


def _start_violation_reporter_once():
    """Start the async DB reporter, including when app.py is loaded by flask run."""
    global _reporter_thread
    if _reporter_thread and _reporter_thread.is_alive():
        return
    with _reporter_lock:
        if _reporter_thread and _reporter_thread.is_alive():
            return
        _reporter_thread = threading.Thread(target=_violation_reporter_worker, daemon=True)
        _reporter_thread.start()


def _queue_violation(plate_text: str, vtype: str):
    """Non-blocking: add a violation to the queue if cooldown has elapsed."""
    _start_violation_reporter_once()
    key = (plate_text, vtype)
    now = time.monotonic()
    if now - _plate_cooldowns.get(key, 0) >= VIOLATION_COOLDOWN_SECS:
        _plate_cooldowns[key] = now
        try:
            _violation_q.put_nowait((plate_text, vtype))
        except queue.Full:
            pass


def _violation_reporter_worker():
    """Background daemon thread: drains CV detections and writes to PostgreSQL."""
    if not _DB_ENABLED:
        return
    with app.app_context():
        from backend.models import User, Violation, ScannerChallan, ScannerChallanItem
        from backend.extensions import db as _db
        while True:
            try:
                plate, vtype = _violation_q.get(timeout=1)
                user = User.query.filter_by(license_plate=plate).first()
                label = _VIOLATION_LABELS.get(vtype, vtype.capitalize())
                if label in _recent_scanner_violation_types(plate):
                    print(f"[DB] Skipped duplicate challan: {label} | plate={plate} | within 24h")
                    continue
                v = Violation(
                    user_id        = user.id if user else None,
                    plate_number   = plate,
                    violation_type = vtype,
                    fine_amount    = _FINE_MAP.get(vtype, 500),
                    status         = "PENDING",
                )
                _db.session.add(v)
                user_info = _MOCK_USERS.get(plate)
                challan_owner = user_info.get("owner") if user_info else (user.name if user else "Unknown")
                challan_vehicle = user_info.get("vehicle") if user_info else (user.vehicle if user else "Unknown")
                challan = ScannerChallan(
                    id           = _next_db_challan_id(),
                    user_id      = user.id if user else None,
                    plate_number = plate,
                    owner_name   = challan_owner,
                    vehicle      = challan_vehicle or "Unknown",
                    fine_amount  = _FINE_MAP.get(vtype, 500),
                    status       = "UNPAID",
                )
                challan.items.append(ScannerChallanItem(
                    type   = label,
                    fine   = _FINE_MAP.get(vtype, 500),
                    source = "violation",
                ))
                _db.session.add(challan)
                _db.session.commit()
                print(f"[DB] Detection saved: {vtype} | plate={plate} | challan={challan.id}")
            except queue.Empty:
                continue
            except Exception as exc:
                print(f"[DB] Error saving violation: {exc}")
                try:
                    _db.session.rollback()
                except Exception:
                    pass


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


def find_motorcycle_for_rider(rider_box: tuple, motorcycles: list):
    """
    Return the motorcycle box that most overlaps rider_box, or None.
    Used to spatially link a no-helmet rider back to their vehicle for
    targeted plate detection.
    """
    px1, py1, px2, py2 = rider_box
    best_box, best_inter = None, 0
    for (mx1, my1, mx2, my2) in motorcycles:
        ix1 = max(px1, mx1);  ix2 = min(px2, mx2)
        iy1 = max(py1, my1);  iy2 = min(py2, my2)
        if ix2 > ix1 and iy2 > iy1:
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter > best_inter:
                best_inter = inter
                best_box   = (mx1, my1, mx2, my2)
    return best_box


# ---------------------------------------------------------------------------
# Per-Vehicle Plate Cache
# ---------------------------------------------------------------------------

class _VehiclePlateCache:
    """
    Caches plate detection results per vehicle position to avoid running
    OCR on every frame for every violating vehicle.

    Vehicles are identified by snapping their centroid to a coarse grid
    (GRID px × GRID px cells), which tolerates small box jitter between
    frames while treating spatially distinct vehicles as separate entries.
    Entries older than MAX_AGE frames are re-queried automatically.
    """

    GRID    = 60   # pixel grid for centroid snapping
    MAX_AGE = 10   # frames before re-running plate OCR for this position

    def __init__(self):
        # grid_key -> (last_frame_count, plate_text, plate_box)
        self._data: dict = {}

    def get_or_refresh(self, vehicle_box: tuple, frame, frame_count: int):
        """
        Return cached (plate_text, plate_box) for vehicle_box, refreshing
        via detect_number_plate if the entry is absent or stale.
        """
        x1, y1, x2, y2 = vehicle_box
        gkey = ((x1 + x2) // 2 // self.GRID,
                (y1 + y2) // 2 // self.GRID)

        entry = self._data.get(gkey)
        if entry is not None:
            last_fc, plate_text, plate_box = entry
            if frame_count - last_fc < self.MAX_AGE:
                return plate_text, plate_box

        plate_text, plate_box = detect_number_plate(frame, region=vehicle_box)
        self._data[gkey] = (frame_count, plate_text, plate_box)
        return plate_text, plate_box

    def clear(self):
        self._data.clear()


_plate_cache = _VehiclePlateCache()

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
# Shared Detection State  (process_frame writes, /detections reads)
# ---------------------------------------------------------------------------
# This is the missing link: detection results were only painted onto pixels
# before; now they are also stored here so the frontend can fetch them as JSON.
_latest_detection = {
    "helmet_violation":   False,
    "triple_violation":   False,
    "overspeed_violation": False,
    "plate_number":       None,
    "violations_active":  [],      # list of active violation type strings
    "timestamp":          0.0,
}
_detection_lock = threading.Lock()

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

def detect_number_plate(frame, region=None):
    """
    Detect and read the number plate closest to *region*.

    Args:
        frame  (np.ndarray): Full BGR video frame.
        region (tuple|None): (x1,y1,x2,y2) bounding box of the violating
                             vehicle.  When given, the plate model searches
                             only within that area (+ 20 % padding), which
                             greatly reduces false positives from unrelated
                             plates elsewhere in the frame.  When None the
                             entire frame is searched (legacy behaviour).

    Returns:
        plate_text (str | None): OCR-extracted plate text, or None.
        plate_box  (tuple | None): Bounding box in full-frame coordinates.
    """
    if _plate_model is None:
        return None, None

    fh, fw = frame.shape[:2]

    if region is not None:
        rx1, ry1, rx2, ry2 = region
        # 20 % padding so the plate is not clipped at vehicle edges
        pad_x = max(10, int((rx2 - rx1) * 0.20))
        pad_y = max(10, int((ry2 - ry1) * 0.20))
        rx1 = max(0,  rx1 - pad_x);  rx2 = min(fw, rx2 + pad_x)
        ry1 = max(0,  ry1 - pad_y);  ry2 = min(fh, ry2 + pad_y)
        search = frame[ry1:ry2, rx1:rx2]
        offset = (rx1, ry1)
    else:
        search = frame
        offset = (0, 0)

    if search.size == 0:
        return None, None

    results = _plate_model(search, verbose=False)[0]
    if not results.boxes:
        return None, None

    best = max(results.boxes, key=lambda b: float(b.conf[0]))
    if float(best.conf[0]) < 0.4:
        return None, None

    lx1, ly1, lx2, ly2 = map(int, best.xyxy[0])
    ox, oy   = offset
    plate_box = (lx1 + ox, ly1 + oy, lx2 + ox, ly2 + oy)

    if _ocr_reader is None:
        return None, plate_box

    # Crop plate from the original full-resolution frame for best OCR quality
    px1, py1, px2, py2 = plate_box
    plate_crop = frame[py1:py2, px1:px2]
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
        any_triple     (bool):        True if a violation was found.
        violating_boxes (list[tuple]): Bounding boxes of motorcycles with
                                       3+ riders — used for plate targeting.
    """
    effective_motos = list(motorcycles)
    if not effective_motos and len(persons) >= TRIPLE_THRESHOLD:
        sx1 = min(p[0] for p in persons);  sx2 = max(p[2] for p in persons)
        sy1 = min(p[1] for p in persons);  sy2 = max(p[3] for p in persons)
        effective_motos = [(sx1, sy1, sx2, sy2)]

    any_triple      = False
    violating_boxes = []

    for (mx1, my1, mx2, my2) in effective_motos:
        rider_count = sum(
            1 for p in persons
            if is_person_on_motorcycle(p, (mx1, my1, mx2, my2))
        )
        triple = rider_count >= TRIPLE_THRESHOLD

        if triple:
            any_triple = True
            violating_boxes.append((mx1, my1, mx2, my2))
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

    return any_triple, violating_boxes


# ---------------------------------------------------------------------------
# Overspeed Module  (Dev 3)
# ---------------------------------------------------------------------------

def detect_overspeed(frame, vehicles):
    """
    Detect ALL visible vehicles exceeding OVERSPEED_LIMIT_KMPH using two trip lines.

    Side-effects: draws trip lines and per-vehicle speed labels onto frame.

    Returns:
        any_overspeed   (bool):        True if ANY vehicle exceeded the limit.
        violating_boxes (list[tuple]): Bounding boxes of currently overspeeding
                                       vehicles — used for plate targeting.
    """
    w = frame.shape[1]

    cv2.line(frame, (0, OVERSPEED_LINE1_Y), (w, OVERSPEED_LINE1_Y), (0, 200, 255), 2)
    cv2.putText(frame, "LINE 1", (10, OVERSPEED_LINE1_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
    cv2.line(frame, (0, OVERSPEED_LINE2_Y), (w, OVERSPEED_LINE2_Y), (255, 200, 0), 2)
    cv2.putText(frame, "LINE 2", (10, OVERSPEED_LINE2_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    matched         = _tracker.update(vehicles)
    any_overspeed   = False
    violating_boxes = []

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
            violating_boxes.append(box)

        if track.speed_kmph is not None:
            color = (0, 0, 255) if track.overspeeding else (0, 220, 0)
            cv2.putText(frame, f"{track.speed_kmph:.1f} km/h",
                        (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif track.crossed_line1:
            elapsed = time.time() - track.start_time
            cv2.putText(frame, f"T{track.id}: {elapsed:.1f}s",
                        (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2)

    return any_overspeed, violating_boxes


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
# Unified Violation Handler
# ---------------------------------------------------------------------------

def process_violation(frame, vehicle_box: tuple, frame_count: int):
    """
    Detect, cache, and annotate the number plate for a single violating vehicle.

    This is the single entry-point for all plate-detection work regardless of
    violation type (helmet, triple riding, overspeed).  It delegates caching
    and throttling to _VehiclePlateCache so each vehicle position is only
    re-queried every MAX_AGE frames, even when multiple violation types fire
    simultaneously on the same vehicle.

    Args:
        frame       : BGR numpy array (modified in-place with plate annotations).
        vehicle_box : (x1,y1,x2,y2) bounding box of the violating vehicle.
        frame_count : Current global frame counter (used for cache expiry).

    Returns:
        plate_text (str | None): OCR result for the caller to include in the
                                 summary overlay, or None if not detected.
    """
    plate_text, plate_box = _plate_cache.get_or_refresh(
        vehicle_box, frame, frame_count
    )

    if plate_box:
        px1, py1, px2, py2 = plate_box
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
        cv2.putText(frame, "Plate", (px1, py1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if plate_text:
            cv2.putText(frame, plate_text, (px1, py2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    return plate_text


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
    Full detection pipeline — all three violation types in one pass.

      Dev 1 — Helmet detection + Number plate OCR
      Dev 2 — Triple riding detection
      Dev 3 — Overspeed detection

    Number plate detection is triggered by ANY violation, always targeted at
    the specific violating vehicle's bounding box via process_violation().
    Modifies frame in-place with annotations and returns it.
    """
    global _frame_count
    _frame_count += 1

    # ── YOLO: persons, motorcycles, cars ─────────────────────────────────────
    persons, motorcycles, cars = run_yolo_detection(frame)
    vehicles   = motorcycles + cars
    riders     = [p for p in persons if is_riding(p, motorcycles)]
    non_riders = [p for p in persons if p not in riders]

    # ── Dev 1: Helmet detection (riders only) ────────────────────────────────
    helmet_results, helmet_violation = detect_helmet(frame, riders)

    # ── Dev 2: Triple riding detection ───────────────────────────────────────
    triple_violation, triple_boxes = detect_triple_riding(frame, persons, motorcycles)

    # ── Dev 3: Overspeed detection ───────────────────────────────────────────
    overspeed_violation, speed_boxes = detect_overspeed(frame, vehicles)

    # ── Collect (vehicle_box, label) for every active violation ──────────────
    # Each entry will be processed by process_violation() below.
    plate_targets = []

    # Helmet — map each no-helmet rider back to their motorcycle
    if helmet_violation:
        for r in helmet_results:
            if not r["has_helmet"]:
                moto = find_motorcycle_for_rider(r["box"], motorcycles)
                if moto:
                    plate_targets.append(moto)

    # Triple riding — motorcycle boxes already returned by detector
    plate_targets.extend(triple_boxes)

    # Overspeed — vehicle boxes already returned by detector
    plate_targets.extend(speed_boxes)

    # ── Plate detection: one call per unique vehicle position, all violations ─
    # process_violation deduplicates via _VehiclePlateCache, so overlapping
    # violations on the same vehicle never trigger redundant OCR inference.
    plate_texts = []
    for vbox in plate_targets:
        text = process_violation(frame, vbox, _frame_count)
        if text:
            plate_texts.append(text)

    # First detected plate is shown in the challan overlay.
    overlay_plate = plate_texts[0] if plate_texts else None

    # ── Draw bounding boxes ───────────────────────────────────────────────────
    draw_boxes(frame, non_riders, color=COLOR_PERSON, label="Person")

    if helmet_results:
        draw_helmet_results(frame, helmet_results)
    else:
        draw_boxes(frame, riders, color=COLOR_PERSON, label="Rider")

    # Motorcycles drawn by detect_triple_riding; draw cars separately.
    draw_boxes(frame, cars, color=(0, 180, 255), label="Car")

    # ── Violation overlay (challan summary box) ───────────────────────────────
    draw_violation_overlay(frame, helmet_violation, triple_violation,
                           overspeed_violation, overlay_plate)

    # ── Publish detection state for /detections endpoint ─────────────────────
    active = []
    if helmet_violation:
        active.append("helmet")
    if triple_violation:
        active.append("triple")
    if overspeed_violation:
        active.append("overspeed")

    with _detection_lock:
        _latest_detection["helmet_violation"]    = helmet_violation
        _latest_detection["triple_violation"]    = triple_violation
        _latest_detection["overspeed_violation"] = overspeed_violation
        _latest_detection["plate_number"]        = overlay_plate
        _latest_detection["violations_active"]   = active
        _latest_detection["timestamp"]           = time.time()

    # ── Queue detected violations for async DB recording ──────────────────────
    if overlay_plate:
        if helmet_violation:
            _queue_violation(overlay_plate, "helmet")
        if triple_violation:
            _queue_violation(overlay_plate, "triple")
        if overspeed_violation:
            _queue_violation(overlay_plate, "overspeed")

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
    _plate_cache.clear()        # discard stale plates from the previous source
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


@app.route("/detections")
def detections():
    """
    Return the latest detection result as JSON.
    The frontend polls this endpoint (every ~1.5 s) to render a live
    violation panel alongside the MJPEG stream.

    Response schema:
    {
        "helmet_violation":    bool,
        "triple_violation":    bool,
        "overspeed_violation": bool,
        "plate_number":        str | null,
        "violations_active":   list[str],   // e.g. ["helmet", "overspeed"]
        "timestamp":           float        // unix epoch of last detection
    }
    """
    with _detection_lock:
        data = dict(_latest_detection)
    return jsonify(data)


# ---------------------------------------------------------------------------
# Smart Traffic Signal System — separate capture pipeline
# ---------------------------------------------------------------------------
_signal_latest_frame  = None
_signal_frame_lock    = threading.Lock()
_signal_worker_source = None   # sentinel: worker exits when this changes


def _signal_capture_worker(source):
    global _signal_latest_frame
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return
    try:
        while _signal_worker_source == source:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.033)
                continue
            frame = cv2.resize(frame, (854, 480))
            frame = _signal_system.process_frame(frame)
            with _signal_frame_lock:
                _signal_latest_frame = frame
    finally:
        cap.release()


def _start_signal_capture(source):
    global _signal_worker_source, _signal_latest_frame
    _signal_worker_source = source
    with _signal_frame_lock:
        _signal_latest_frame = None
    t = threading.Thread(target=_signal_capture_worker, args=(source,), daemon=True)
    t.start()


def _generate_signal_frames():
    while True:
        with _signal_frame_lock:
            frame = _signal_latest_frame
        if frame is None:
            time.sleep(0.02)
            continue
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if ret:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.033)


@app.route("/signal_video")
def signal_video():
    return Response(
        _generate_signal_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control":     "no-cache, no-store, must-revalidate",
            "Pragma":            "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/signal_status")
def signal_status_route():
    return jsonify(_signal_system.state)


@app.route("/signal_use_webcam", methods=["POST"])
def signal_use_webcam():
    _start_signal_capture(0)
    return jsonify({"success": True})


@app.route("/signal_upload", methods=["POST"])
def signal_upload():
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No file in request."}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Unsupported format."}), 400
    original = secure_filename(file.filename)
    ext      = original.rsplit(".", 1)[1].lower()
    path     = os.path.join(UPLOAD_FOLDER, f"signal_video.{ext}")
    file.save(path)
    _start_signal_capture(path)
    return jsonify({"success": True, "filename": original})


# ---------------------------------------------------------------------------
# Smart Police Scanner + Auto Challan System
# ---------------------------------------------------------------------------

_DEMO_USERS = {
    "KA01AB1234": {"owner": "Raghav Dhingra", "vehicle": "Bike",  "rc": True,  "insurance": False, "puc": True,  "challans": []},
    "MH02CD5678": {"owner": "Arjun Mehta",    "vehicle": "Car",   "rc": True,  "insurance": True,  "puc": False, "challans": []},
    "DL03EF9012": {"owner": "Priya Sharma",   "vehicle": "Truck", "rc": True,  "insurance": False, "puc": False, "challans": []},
    "TN04GH3456": {"owner": "Karthik Raj",    "vehicle": "Car",   "rc": True,  "insurance": True,  "puc": True,  "challans": []},
    "GJ05IJ7890": {"owner": "Sneha Patel",    "vehicle": "Bike",  "rc": False, "insurance": False, "puc": True,  "challans": []},
}


def _load_mock_users() -> dict:
    """
    Load vehicle records from users.json and keep the demo plates available.
    The scanner UI placeholder and quick-fill buttons depend on the demo
    records, while generated data in users.json remains the primary dataset.
    """
    import json as _json
    users_path = os.path.join(os.path.dirname(__file__), "users.json")
    records = dict(_DEMO_USERS)
    if os.path.exists(users_path):
        with open(users_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        records.update(data)
        print(f"[scanner] Loaded {len(data)} vehicle records from users.json (+ {len(_DEMO_USERS)} demo plates)")
        return records
    print("[scanner] users.json not found — using 5 hardcoded records. Run generate_mock_data.py to expand.")
    return records

_MOCK_USERS: dict = _load_mock_users()

_SCANNER_CHALLANS: list = []          # fallback only when PostgreSQL is unavailable
_CHALLAN_COUNTER:  list = [1]         # fallback id counter for in-memory challans
_CHALLAN_LOCK             = threading.Lock()

_DOC_FINES       = {"insurance": 1000, "puc": 500}
_VIOLATION_FINES = {"helmet": 500, "triple": 1000, "overspeed": 1000}
_CHALLAN_DUPLICATE_WINDOW = timedelta(hours=24)


def _now_ist_iso() -> str:
    return datetime.now(IST).replace(microsecond=0).isoformat()


def _parse_challan_timestamp(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def _db_scanner_challans_available() -> bool:
    return (
        _DB_ENABLED
        and _DBUser is not None
        and _DBScannerChallan is not None
        and _DBScannerChallanItem is not None
    )


def _list_db_challans(plate=None) -> list:
    if not _db_scanner_challans_available():
        return []

    try:
        query = _DBScannerChallan.query
        if plate:
            query = query.filter_by(plate_number=plate)
        return [
            challan.to_dict()
            for challan in query.order_by(_DBScannerChallan.created_at.desc()).all()
        ]
    except Exception as exc:
        print(f"[scanner] PostgreSQL challan read failed ({exc})")
        return []


def _next_db_challan_id() -> str:
    latest = (
        _DBScannerChallan.query
        .filter(_DBScannerChallan.id.like("CH%"))
        .order_by(_DBScannerChallan.id.desc())
        .first()
    )

    if latest:
        try:
            return f"CH{int(latest.id[2:]) + 1:04d}"
        except (TypeError, ValueError):
            pass

    return "CH0001"


def _next_memory_challan_id() -> str:
    """Use a separate prefix so fallback challans never collide with DB rows."""
    challan_id = f"TMP{_CHALLAN_COUNTER[0]:04d}"
    _CHALLAN_COUNTER[0] += 1
    return challan_id


def _plate_matches(challan_plate, requested_plate) -> bool:
    return not requested_plate or (challan_plate or "").upper() == requested_plate.upper()


def _find_scanner_challan(challan_id: str, plate=None):
    requested_plate = plate.upper().strip() if plate else None

    with _CHALLAN_LOCK:
        for challan in _SCANNER_CHALLANS:
            if challan["id"] == challan_id and _plate_matches(challan.get("plate"), requested_plate):
                return "memory", challan

    if _db_scanner_challans_available():
        try:
            challan = db.session.get(_DBScannerChallan, challan_id)
            if challan and _plate_matches(challan.plate_number, requested_plate):
                return "db", challan
        except Exception as exc:
            print(f"[scanner] PostgreSQL challan lookup failed ({exc})")

    return None, None


def _scanner_payment_client():
    if not _RAZORPAY_AVAILABLE:
        return None
    key_id = app.config.get("RAZORPAY_KEY_ID", "")
    key_secret = app.config.get("RAZORPAY_KEY_SECRET", "")
    if not key_id or not key_secret:
        return None
    return _razorpay_lib.Client(auth=(key_id, key_secret))


def _recent_scanner_violation_types(plate: str) -> set:
    cutoff_utc = datetime.utcnow() - _CHALLAN_DUPLICATE_WINDOW
    recent_types = set()

    if _db_scanner_challans_available():
        try:
            rows = (
                _DBScannerChallanItem.query
                .join(_DBScannerChallan)
                .filter(_DBScannerChallan.plate_number == plate)
                .filter(_DBScannerChallan.created_at >= cutoff_utc)
                .all()
            )
            recent_types.update(item.type for item in rows)
        except Exception as exc:
            print(f"[scanner] Duplicate challan check failed ({exc})")

    cutoff_ist = datetime.now(IST) - _CHALLAN_DUPLICATE_WINDOW
    with _CHALLAN_LOCK:
        for challan in _SCANNER_CHALLANS:
            if challan.get("plate") != plate:
                continue
            created_at = _parse_challan_timestamp(challan.get("timestamp"))
            if created_at and created_at >= cutoff_ist:
                recent_types.update(item.get("type") for item in challan.get("violations", []))

    return {v for v in recent_types if v}


def _scanner_payload(plate: str, user: dict, include_challans: bool = False, source: str = "users.json") -> dict:
    payload = {
        "found":    True,
        "plate":    plate,
        "owner":    user["owner"],
        "vehicle":  user["vehicle"],
        "documents": {
            "rc":        user["rc"],
            "insurance": user["insurance"],
            "puc":       user["puc"],
        },
        "source": source,
    }

    if include_challans:
        db_challans = _list_db_challans(plate)
        if db_challans:
            payload["challans"] = db_challans
        else:
            with _CHALLAN_LOCK:
                payload["challans"] = [c for c in _SCANNER_CHALLANS if c["plate"] == plate]

    return payload


def _lookup_scanner_user(plate: str):
    mock_user = _MOCK_USERS.get(plate)
    if mock_user:
        return mock_user, None, "users.json"

    if _DB_ENABLED and _DBUser is not None:
        try:
            db_user = _DBUser.query.filter_by(license_plate=plate).first()
            if db_user is not None:
                return {
                    "owner":     db_user.name,
                    "vehicle":   db_user.vehicle or "Unknown",
                    "rc":        db_user.rc,
                    "insurance": db_user.insurance,
                    "puc":       db_user.puc,
                }, db_user, "postgresql"
        except Exception as exc:
            print(f"[scanner] PostgreSQL user lookup failed ({exc})")

    return None, None, None


@app.route("/scan/<plate>")
def scan_plate(plate):
    """
    Look up a vehicle by plate number.

    The scanner now checks users.json first, because the React scanner,
    user challan page, quick-fill buttons, and challan creation all use that
    same in-memory dataset. PostgreSQL remains a secondary fallback when it is
    configured and seeded.
    """
    plate = plate.upper().strip()
    print(f"[scan] Query: {plate}")

    user, _db_user, source = _lookup_scanner_user(plate)
    if user:
        print(f"[scan] FOUND in {source}: {plate} -> {user['owner']} ({user['vehicle']})")
        return jsonify(_scanner_payload(plate, user, source=source))

    print(f"[scan] NOT FOUND: {plate}")
    return jsonify({
        "found":   False,
        "message": f"No vehicle registered for plate {plate}. Try one of the quick-fill sample plates.",
    }), 404


@app.route("/create-challan", methods=["POST"])
def create_challan():
    data                = request.get_json(force=True)
    plate               = data.get("plate", "").upper().strip()
    selected_violations = data.get("selected_violations", [])

    user, db_user, source = _lookup_scanner_user(plate)
    if not user:
        return jsonify({"success": False, "error": f"Vehicle not found: {plate}"}), 404

    violations, total_fine = [], 0

    # Auto-add document violations
    doc_map = {"insurance": user["insurance"], "puc": user["puc"]}
    for doc, valid in doc_map.items():
        if not valid:
            fine = _DOC_FINES[doc]
            violations.append({"type": f"No {doc.upper()}", "fine": fine, "source": "document"})
            total_fine += fine

    # Add officer-selected violations
    for v in selected_violations:
        fine = _VIOLATION_FINES.get(v, 0)
        if fine:
            label = {"helmet": "No Helmet", "triple": "Triple Riding", "overspeed": "Overspeed"}.get(v, v.capitalize())
            violations.append({"type": label, "fine": fine, "source": "violation"})
            total_fine += fine

    recent_types = _recent_scanner_violation_types(plate)
    skipped = [item["type"] for item in violations if item["type"] in recent_types]
    violations = [item for item in violations if item["type"] not in recent_types]
    total_fine = sum(item["fine"] for item in violations)

    if not violations:
        return jsonify({
            "success": False,
            "error": "This vehicle already has challans for these violation type(s) in the last 24 hours.",
            "skipped": skipped,
        }), 409

    if _db_scanner_challans_available():
        try:
            challan = _DBScannerChallan(
                id           = _next_db_challan_id(),
                user_id      = db_user.id if db_user else None,
                plate_number = plate,
                owner_name   = user["owner"],
                vehicle      = user["vehicle"],
                fine_amount  = total_fine,
                status       = "UNPAID",
            )
            for item in violations:
                challan.items.append(_DBScannerChallanItem(
                    type   = item["type"],
                    fine   = item["fine"],
                    source = item["source"],
                ))
            db.session.add(challan)
            db.session.commit()
            return jsonify({"success": True, "challan": challan.to_dict(), "skipped": skipped})
        except Exception as exc:
            db.session.rollback()
            print(f"[scanner] PostgreSQL challan write failed ({exc}); using memory fallback.")

    with _CHALLAN_LOCK:
        challan = {
            "id":         _next_memory_challan_id(),
            "plate":      plate,
            "owner":      user["owner"],
            "vehicle":    user["vehicle"],
            "violations": violations,
            "fine":       total_fine,
            "status":     "UNPAID",
            "timestamp":  _now_ist_iso(),
        }

        _SCANNER_CHALLANS.append(challan)

    return jsonify({"success": True, "challan": challan, "skipped": skipped})


@app.route("/user/<plate>")
def get_user_challans(plate):
    """
    Single endpoint used by BOTH dashboards:
      - User Dashboard  → reads owner/vehicle/challans
      - Police Scanner  → also reads documents (rc, insurance, puc, dl)
    Source: _MOCK_USERS (in-memory dict loaded from users.json at startup).
    """
    plate = plate.upper().strip()
    user, _db_user, source = _lookup_scanner_user(plate)

    if not user:
        return jsonify({"found": False, "challans": [],
                        "message": f"No vehicle registered for {plate}"}), 404

    return jsonify(_scanner_payload(plate, user, include_challans=True, source=source))


@app.route("/pay/<challan_id>", methods=["POST"])
def pay_challan(challan_id):
    data = request.get_json(silent=True) or {}
    plate = data.get("plate")
    source, challan = _find_scanner_challan(challan_id, plate)

    if source == "db":
        try:
            if challan.status != "PAID":
                challan.status = "PAID"
                db.session.commit()
            return jsonify({"success": True, "challan": challan.to_dict()})
        except Exception as exc:
            db.session.rollback()
            print(f"[scanner] PostgreSQL challan payment failed ({exc})")
            return jsonify({"success": False, "error": "Payment failed"}), 500

    if source == "memory":
        with _CHALLAN_LOCK:
            if challan["status"] != "PAID":
                challan["status"] = "PAID"
            return jsonify({"success": True, "challan": challan})

    return jsonify({"success": False, "error": "Challan not found"}), 404


@app.route("/payments/create-scanner-order", methods=["POST"])
def create_scanner_order():
    data = request.get_json(silent=True) or {}
    challan_id = data.get("challan_id", "")
    plate = data.get("plate")
    source, challan = _find_scanner_challan(challan_id, plate)

    if not challan:
        return jsonify({"error": "Challan not found"}), 404

    status = challan.status if source == "db" else challan["status"]
    if status == "PAID":
        return jsonify({"error": "Already paid"}), 400

    amount = challan.fine_amount if source == "db" else challan["fine"]
    client = _scanner_payment_client()
    if not client:
        return jsonify({"error": "Payment gateway not configured — add Razorpay keys to .env"}), 503

    try:
        order = client.order.create({
            "amount": amount * 100,
            "currency": "INR",
            "receipt": f"scanner_{challan_id}",
            "notes": {
                "challan_id": challan_id,
                "plate": plate or (challan.plate_number if source == "db" else challan["plate"]),
            },
        })
    except Exception as exc:
        print(f"[scanner] Razorpay order creation failed ({exc})")
        return jsonify({"error": "Could not open payment gateway. Please try again."}), 502

    return jsonify({
        "order_id": order["id"],
        "amount": amount,
        "currency": "INR",
        "key_id": app.config.get("RAZORPAY_KEY_ID"),
        "challan": {
            "id": challan_id,
            "plate": plate or (challan.plate_number if source == "db" else challan["plate"]),
        },
    })


@app.route("/payments/verify-scanner", methods=["POST"])
def verify_scanner_payment():
    data = request.get_json(silent=True) or {}
    order_id = data.get("razorpay_order_id", "")
    payment_id = data.get("razorpay_payment_id", "")
    signature = data.get("razorpay_signature", "")
    challan_id = data.get("challan_id", "")
    plate = data.get("plate")

    key_secret = app.config.get("RAZORPAY_KEY_SECRET", "")
    message = f"{order_id}|{payment_id}"
    expected = hmac.new(
        key_secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if expected != signature:
        return jsonify({"error": "Invalid payment signature"}), 400

    source, challan = _find_scanner_challan(challan_id, plate)
    if not challan:
        return jsonify({"error": "Challan not found"}), 404

    if source == "db":
        challan.status = "PAID"
        db.session.commit()
        return jsonify({"success": True, "challan": challan.to_dict()})

    with _CHALLAN_LOCK:
        challan["status"] = "PAID"
        return jsonify({"success": True, "challan": challan})


@app.route("/all-challans")
def all_challans():
    db_challans = _list_db_challans()
    if db_challans:
        return jsonify({"challans": db_challans})

    with _CHALLAN_LOCK:
        challans = list(_SCANNER_CHALLANS)
    return jsonify({"challans": challans})


@app.route("/sample-plates")
def sample_plates():
    """
    Return up to `n` representative plates from the scanner dataset.
    Prioritises vehicles with at least one missing document so quick-fill
    buttons in the UI demonstrate violations automatically.
    """
    n = min(int(request.args.get("n", 8)), 50)

    rows = list(_MOCK_USERS.items())
    violators = [
        (plate, user) for plate, user in rows
        if not all(user.get(doc, False) for doc in ("rc", "insurance", "puc"))
    ]
    clean = [
        (plate, user) for plate, user in rows
        if all(user.get(doc, False) for doc in ("rc", "insurance", "puc"))
    ]

    random.shuffle(violators)
    random.shuffle(clean)
    chosen = violators[:max(0, n - 1)] + clean[:max(1, n - len(violators))]
    random.shuffle(chosen)

    return jsonify([
        {
            "plate":   plate,
            "owner":   user["owner"],
            "vehicle": user.get("vehicle") or "Unknown",
            "issues":  [d.upper() for d in ("rc", "insurance", "puc")
                        if not user.get(d, False)],
        }
        for plate, user in chosen[:n]
    ])


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialise DB tables (no-op if they already exist)
    if _DB_ENABLED:
        with app.app_context():
            try:
                db.create_all()
                print("[DB] Tables verified/created.")
            except Exception as exc:
                print(f"[DB] Could not create tables ({exc}). Is PostgreSQL running?")

    # Start async violation reporter
    _start_violation_reporter_once()

    _start_capture(0)                               # violation detection — webcam
    _start_signal_capture(0)                        # signal system — webcam
    app.run(host="0.0.0.0", port=9000, debug=False, threaded=True)
