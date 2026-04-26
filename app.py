"""
Smart Traffic Violation Detection System
=========================================
Team: 3 Developers
- Developer 1: Helmet Detection + Number Plate Detection + OCR
- Developer 2: Triple Riding Detection
- Developer 3: Overspeed Detection
"""

import os
import time
import math
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

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

# YOLO class IDs (COCO dataset)
CLASS_PERSON     = 0
CLASS_MOTORCYCLE = 3
CLASS_CAR        = 2

# ---------------------------------------------------------------------------
# Overspeed Constants (Dev 3)
# ---------------------------------------------------------------------------
OVERSPEED_LINE1_Y      = 160    # upper trip line y-coordinate
OVERSPEED_LINE2_Y      = 420    # lower trip line y-coordinate
OVERSPEED_REAL_DIST    = 10.0   # real-world distance between lines (metres)
OVERSPEED_LIMIT_KMPH   = 40.0   # speed threshold in km/h
OVERSPEED_MISSING_MAX  = 30     # frames before state auto-resets (higher = more tolerant)
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
        self.tracks   = {}   # track_id -> _VehicleTrack
        self._next_id = 0

    def reset(self):
        self.tracks   = {}
        self._next_id = 0

    def update(self, boxes):
        """
        Match boxes to existing tracks by nearest centroid.
        Create new tracks for unmatched boxes.
        Evict tracks that have been missing too long.

        Args:
            boxes: [(x1,y1,x2,y2), ...]

        Returns:
            [(box, track), ...] for every currently active detection.
        """
        centroids    = [((x1+x2)//2, (y1+y2)//2) for x1,y1,x2,y2 in boxes]
        box_to_track = {}
        used_boxes   = set()

        # ── Match existing tracks to nearest unmatched detection ──────────────
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

        # ── Create new tracks for unmatched detections ────────────────────────
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
_latest_frame  = None           # most recent processed frame
_frame_lock    = threading.Lock()
_worker_source = None           # sentinel: worker exits when this changes

# ---------------------------------------------------------------------------
# Helmet Module (Dev 1)
# ---------------------------------------------------------------------------

def detect_helmet(frame, persons):
    """
    Detect whether riders are wearing helmets.

    Args:
        frame      (np.ndarray): Current video frame (BGR).
        persons    (list[tuple]): Bounding boxes [(x1,y1,x2,y2), ...] of detected persons.

    Returns:
        bool: True if any person is detected without a helmet, False otherwise.
    """
    # TODO (Developer 1): Implement helmet detection logic here.
    # Suggested approach:
    #   - Crop each person ROI from frame.
    #   - Run a secondary classifier / YOLO model on the ROI.
    #   - Return True if at least one person has no helmet.
    return False


# ---------------------------------------------------------------------------
# Plate Module (Dev 1)
# ---------------------------------------------------------------------------

def detect_number_plate(frame):
    """
    Detect and read the number plate from the frame.

    Args:
        frame (np.ndarray): Current video frame (BGR).

    Returns:
        tuple:
            plate_text (str | None): OCR-extracted plate text, or None if not found.
            plate_box  (tuple | None): Bounding box (x1,y1,x2,y2) of the plate, or None.
    """
    # TODO (Developer 1): Implement number plate detection + OCR here.
    # Suggested approach:
    #   - Use a plate-detection YOLO model or OpenCV contour methods.
    #   - Run EasyOCR / pytesseract on the cropped plate ROI.
    #   - Return the plate text and its bounding box.
    return None, None


# ---------------------------------------------------------------------------
# Triple Riding Module (Dev 2)
# ---------------------------------------------------------------------------

def detect_triple_riding(frame, persons, motorcycles):
    """
    Detect triple riding (3+ persons on a single motorcycle).

    Args:
        frame       (np.ndarray): Current video frame (BGR).
        persons     (list[tuple]): Bounding boxes of detected persons.
        motorcycles (list[tuple]): Bounding boxes of detected motorcycles.

    Returns:
        bool: True if triple riding is detected, False otherwise.
    """
    # TODO (Developer 2): Implement triple riding detection logic here.
    # Suggested approach:
    #   - For each motorcycle box, count how many person boxes overlap (IoU / containment).
    #   - If overlap count >= 3 for any motorcycle, return True.
    return False


# ---------------------------------------------------------------------------
# Overspeed Module (Dev 3)
# ---------------------------------------------------------------------------

def detect_overspeed(frame, vehicles):
    """
    Detect ALL visible vehicles exceeding OVERSPEED_LIMIT_KMPH using two trip lines.

    Multi-vehicle tracking
    ----------------------
    Every detected vehicle gets its own independent timer via _MultiVehicleTracker.
    Nearest-centroid matching associates each box to its track across frames.
    Speed is computed per-vehicle when its centroid crosses LINE2.

    Side-effects: draws trip lines and per-vehicle speed labels onto `frame`.

    Args:
        frame    (np.ndarray): Current video frame (BGR).
        vehicles (list[tuple]): Bounding boxes [(x1,y1,x2,y2), ...].

    Returns:
        bool: True if ANY tracked vehicle exceeded OVERSPEED_LIMIT_KMPH.
    """
    w = frame.shape[1]

    # ── Trip lines (always drawn) ──────────────────────────────────────────────
    cv2.line(frame, (0, OVERSPEED_LINE1_Y), (w, OVERSPEED_LINE1_Y), (0, 200, 255), 2)
    cv2.putText(frame, "LINE 1", (10, OVERSPEED_LINE1_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
    cv2.line(frame, (0, OVERSPEED_LINE2_Y), (w, OVERSPEED_LINE2_Y), (255, 200, 0), 2)
    cv2.putText(frame, "LINE 2", (10, OVERSPEED_LINE2_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    # ── Update multi-vehicle tracker ──────────────────────────────────────────
    matched      = _tracker.update(vehicles)
    any_overspeed = False

    for box, track in matched:
        x1, y1, x2, y2 = box
        cx, cy          = track.last_cx, track.last_cy

        # Centroid dot
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        # ── Per-track timing ──────────────────────────────────────────────────
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

        # ── Per-vehicle speed label ───────────────────────────────────────────
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
    results = model(frame, verbose=False)[0]

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


def draw_violation_overlay(frame, helmet_violation, triple_violation,
                           overspeed_violation, plate_text):
    h, w = frame.shape[:2]
    y_offset = 30
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.75
    thickness = 2

    if helmet_violation:
        cv2.putText(frame, "No Helmet X", (10, y_offset),
                    font, scale, (0, 0, 255), thickness)
        y_offset += 35

    if triple_violation:
        cv2.putText(frame, "Triple Riding !!!", (10, y_offset),
                    font, scale, (0, 69, 255), thickness)
        y_offset += 35

    if overspeed_violation:
        cv2.putText(frame, "Overspeed >>>", (10, y_offset),
                    font, scale, (0, 165, 255), thickness)
        y_offset += 35

    if helmet_violation or triple_violation or overspeed_violation:
        plate_display = plate_text if plate_text else "Detecting..."
        cv2.putText(frame, f"Plate: {plate_display}", (10, y_offset),
                    font, 0.65, (255, 255, 0), thickness)
        y_offset += 30
        fine = "Fine: Rs.1000" if (triple_violation or overspeed_violation) else "Fine: Rs.500"
        cv2.putText(frame, fine, (10, y_offset),
                    font, 0.65, (255, 255, 0), thickness)


# ---------------------------------------------------------------------------
# Main Frame-Processing Pipeline
# ---------------------------------------------------------------------------

def process_frame(frame):
    """
    Full pipeline executed for every captured frame.
    """
    # --- YOLO Detection ---
    persons, motorcycles, cars = run_yolo_detection(frame)
    vehicles = motorcycles + cars

    # --- Module Calls ---

    # Helmet Module (Dev 1)
    helmet_violation = detect_helmet(frame, persons)

    # Triple Riding Module (Dev 2)
    triple_violation = detect_triple_riding(frame, persons, motorcycles)

    # Overspeed Module (Dev 3)
    overspeed_violation = detect_overspeed(frame, vehicles)

    # --- Unified Violation Flag ---
    violation = helmet_violation or triple_violation or overspeed_violation

    # --- Plate Detection (only on violation) ---
    plate_text, plate_box = None, None
    if violation:
        plate_text, plate_box = detect_number_plate(frame)

    # --- Draw Bounding Boxes ---
    draw_boxes(frame, persons,     color=(255, 200, 0),  label="Person")
    draw_boxes(frame, motorcycles, color=(0, 255, 100),  label="Motorcycle")
    draw_boxes(frame, cars,        color=(0, 180, 255),  label="Car")
    if plate_box:
        draw_boxes(frame, [plate_box], color=(0, 255, 255), label="Plate")

    # --- Violation Overlay ---
    draw_violation_overlay(frame, helmet_violation, triple_violation,
                           overspeed_violation, plate_text)

    return frame


# ---------------------------------------------------------------------------
# Background Capture Worker
# ---------------------------------------------------------------------------

def _capture_worker(source):
    """
    Runs in a daemon thread. Captures frames, runs the full YOLO pipeline,
    and stores the latest result in _latest_frame.

    Exits automatically when _worker_source changes (i.e. source switches).
    """
    global _latest_frame

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return

    consecutive_errors = 0
    try:
        while _worker_source == source:         # exit when source is swapped
            ret, frame = cap.read()

            if not ret:
                if isinstance(source, str):     # video file ended → loop
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    _tracker.reset()
                    consecutive_errors = 0
                    continue
                # Webcam MSMF transient error — retry instead of killing the stream
                consecutive_errors += 1
                if consecutive_errors > 60:     # ~2 s of continuous failure
                    break
                time.sleep(0.033)
                continue

            consecutive_errors = 0              # successful read — clear error count

            # Resize before YOLO — faster inference, same line positions
            frame = cv2.resize(frame, (854, 480))
            frame = process_frame(frame)

            with _frame_lock:
                _latest_frame = frame
    finally:
        cap.release()


def _start_capture(source):
    """Switch to a new source: update the sentinel and spawn a fresh daemon thread."""
    global _worker_source, _latest_frame

    _worker_source = source             # old worker sees mismatch → exits
    with _frame_lock:
        _latest_frame = None            # clear stale frame while new thread warms up

    t = threading.Thread(target=_capture_worker, args=(source,), daemon=True)
    t.start()


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

        time.sleep(0.033)               # cap stream at ~30 fps


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control":    "no-cache, no-store, must-revalidate",
            "Pragma":           "no-cache",
            "X-Accel-Buffering":"no",   # disable Nginx/proxy buffering
        }
    )


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
