"""
Smart Traffic Violation Detection System
=========================================
Team: 3 Developers
- Developer 1: Helmet Detection + Number Plate Detection + OCR
- Developer 2: Triple Riding Detection
- Developer 3: Overspeed Detection
"""

import re
import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, render_template
from ultralytics import YOLO
from helmet_detector import HelmetDetector, draw_helmet_results

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------------------------
model = YOLO("yolov8n.pt")

# Helmet detection model (custom YOLOv8 trained for helmet detection)
HELMET_MODEL_PATH = "helmet_model.pt"

try:
    _helmet_detector = HelmetDetector(HELMET_MODEL_PATH, helmet_class=0)
    print(f"[helmet] ✓ Model loaded successfully from {HELMET_MODEL_PATH}")
    print(f"[helmet] Using class index 0 for 'helmet' detection")
except FileNotFoundError:
    print(f"[helmet] ✗ Model file not found at {HELMET_MODEL_PATH}; helmet detection disabled.")
    _helmet_detector = None
except Exception as exc:
    print(f"[helmet] ✗ Model failed to load ({exc}); helmet detection disabled.")
    _helmet_detector = None

# License-plate detection model (custom YOLOv8, e.g. best.pt from Roboflow).
PLATE_MODEL_PATH = "license_plate_detector.pt"

try:
    _plate_model = YOLO(PLATE_MODEL_PATH)
except Exception as exc:
    print(f"[plate] model not loaded ({exc}); plate detection disabled.")
    _plate_model = None

# EasyOCR reader — initialised once; downloads models (~300 MB) on first run.
try:
    _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
except Exception as exc:
    print(f"[plate] EasyOCR not loaded ({exc}); OCR disabled.")
    _ocr_reader = None

# YOLO class IDs (COCO dataset)
CLASS_PERSON     = 0
CLASS_MOTORCYCLE = 3
CLASS_CAR        = 2

# Minimum fraction of a person's area that must overlap a motorcycle box
# for that person to be considered a rider (~15% overlap is enough).
_RIDING_OVERLAP_THRESH = 0.15

# Plate detection is throttled: OCR runs at most once every N frames.
_PLATE_REFRESH_INTERVAL = 10

# Module-level state for plate caching and frame counting.
_frame_count       = 0
_cached_plate_text = None
_cached_plate_box  = None


def is_riding(person_box: tuple, motorcycle_boxes: list) -> bool:
    """
    Return True if *person_box* overlaps a motorcycle by at least
    _RIDING_OVERLAP_THRESH of the person's own area.

    We compare intersection / person_area (not IoU) because a rider's
    body extends well above the motorcycle, so the union is large and
    IoU stays deceptively low even for a clear spatial match.
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
# Helmet Module (Dev 1)
# ---------------------------------------------------------------------------

def detect_helmet(frame, persons):
    """
    Detect whether each person is wearing a helmet.

    Args:
        frame   (np.ndarray)  : Current video frame (BGR).
        persons (list[tuple]) : Person bounding boxes [(x1,y1,x2,y2), ...].

    Returns:
        results       (list[dict]): Per-person dicts with keys
                                    'id', 'box', 'has_helmet'.
                                    Empty list when the model is unavailable.
        any_violation (bool)      : True if at least one person has no helmet.
    """
    if _helmet_detector is None:
        return [], False
    return _helmet_detector.detect(frame, persons)


# ---------------------------------------------------------------------------
# Plate Module (Dev 1)
# ---------------------------------------------------------------------------

def detect_number_plate(frame):
    """
    Detect and read the number plate from the frame.

    Args:
        frame (np.ndarray): Current video frame (BGR).

    Returns:
        plate_text (str | None): OCR-extracted plate text, or None if not found.
        plate_box  (tuple | None): Bounding box (x1,y1,x2,y2) of the plate, or None.
    """
    if _plate_model is None:
        return None, None

    # Step 1: Run plate detection model on the full frame
    results = _plate_model(frame, verbose=False)[0]
    if not results.boxes:
        return None, None

    # Step 2: Pick the single most-confident detection
    best = max(results.boxes, key=lambda b: float(b.conf[0]))
    if float(best.conf[0]) < 0.4:
        return None, None

    x1, y1, x2, y2 = map(int, best.xyxy[0])
    plate_box = (x1, y1, x2, y2)

    if _ocr_reader is None:
        return None, plate_box

    # Step 3: Crop the plate region from the frame
    plate_crop = frame[y1:y2, x1:x2]
    if plate_crop.size == 0:
        return None, plate_box

    # Step 4: Run EasyOCR on the cropped plate image
    ocr_results = _ocr_reader.readtext(plate_crop)
    if not ocr_results:
        return None, plate_box

    # Step 5: Return the text with the highest OCR confidence, stripped of
    # spaces and non-alphanumeric characters so the display is clean.
    best_ocr = max(ocr_results, key=lambda r: r[2])
    plate_text = re.sub(r"[^A-Z0-9]", "", best_ocr[1].upper())

    return plate_text, plate_box


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
    Detect vehicles exceeding the speed threshold.

    Args:
        frame    (np.ndarray): Current video frame (BGR).
        vehicles (list[tuple]): Bounding boxes of detected vehicles (motorcycles + cars).

    Returns:
        bool: True if any vehicle is over the speed limit, False otherwise.
    """
    # TODO (Developer 3): Implement overspeed detection logic here.
    # Suggested approach:
    #   - Track vehicle centroids across frames (e.g., using a simple centroid tracker).
    #   - Estimate pixel displacement per frame and convert to km/h using a known scale factor.
    #   - Return True if any vehicle's estimated speed exceeds the threshold.
    return False


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
        if conf < 0.4:
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
        "/System/Library/Fonts/Supplemental/Arial.ttf",   # macOS
        "/System/Library/Fonts/Helvetica.ttc",             # macOS alt
        "/Library/Fonts/Arial.ttf",                        # macOS user fonts
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()

# Loaded once on first use to avoid repeated disk access
_overlay_font = None


def draw_violation_overlay(frame, helmet_violation, triple_violation,
                           overspeed_violation, plate_text):
    """
    Draw a challan info box using PIL so Unicode characters (❌, ₹) render correctly.
    Modifies *frame* in-place.
    """
    global _overlay_font

    any_violation = helmet_violation or triple_violation or overspeed_violation
    if not any_violation:
        return

    # Build lines: (text, RGB colour)
    lines = []
    if helmet_violation:
        lines.append(("No Helmet ❌", (255, 80, 80)))      # ❌
    if triple_violation:
        lines.append(("Triple Riding !!!", (255, 120, 0)))
    if overspeed_violation:
        lines.append(("Overspeed >>>", (255, 165, 0)))

    plate_display = plate_text if plate_text else "Detecting..."
    lines.append((f"Plate: {plate_display}", (255, 220, 0)))

    fine_amt = "₹1000" if (triple_violation or overspeed_violation) else "₹500"  # ₹
    lines.append((f"Fine: {fine_amt}", (255, 220, 0)))

    # Lazy-load font
    if _overlay_font is None:
        _overlay_font = _get_pil_font(22)

    padding, line_h = 10, 30
    box_w = 280
    box_h = len(lines) * line_h + padding * 2

    # Convert BGR frame → PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)

    # Dark background box
    draw.rectangle([8, 8, 8 + box_w, 8 + box_h], fill=(15, 15, 15))

    # Text lines
    for i, (text, colour) in enumerate(lines):
        draw.text(
            (8 + padding, 8 + padding + i * line_h),
            text, fill=colour, font=_overlay_font,
        )

    # Write result back into frame in-place
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Main Frame-Processing Pipeline
# ---------------------------------------------------------------------------

def process_frame(frame):
    """
    Full pipeline executed for every captured frame.
    """
    global _frame_count, _cached_plate_text, _cached_plate_box
    _frame_count += 1

    # --- YOLO Detection ---
    persons, motorcycles, cars = run_yolo_detection(frame)
    vehicles = motorcycles + cars

    # --- Filter: only persons overlapping a motorcycle are checked for helmets ---
    riders = [p for p in persons if is_riding(p, motorcycles)]

    # --- Module Calls ---

    # Helmet Module (Dev 1) — riders only, not all pedestrians
    helmet_results, helmet_violation = detect_helmet(frame, riders)

    # Triple Riding Module (Dev 2)
    triple_violation = detect_triple_riding(frame, persons, motorcycles)

    # Overspeed Module (Dev 3)
    overspeed_violation = detect_overspeed(frame, vehicles)

    # --- Unified Violation Flag ---
    violation = helmet_violation or triple_violation or overspeed_violation

    # --- Plate Detection (only on violation, throttled to every N frames) ---
    if violation:
        if _frame_count % _PLATE_REFRESH_INTERVAL == 0:
            _cached_plate_text, _cached_plate_box = detect_number_plate(frame)
        plate_text = _cached_plate_text
        plate_box  = _cached_plate_box
    else:
        # Clear cache when no violation so stale data never bleeds into the
        # next violation window.
        _cached_plate_text = None
        _cached_plate_box  = None
        plate_text = None
        plate_box  = None

    # --- Draw Bounding Boxes ---
    # Non-rider persons get a plain yellow box; riders get Helmet/No Helmet label.
    non_riders = [p for p in persons if not is_riding(p, motorcycles)]
    draw_boxes(frame, non_riders, color=(255, 200, 0), label="Person")

    if helmet_results:
        draw_helmet_results(frame, helmet_results)
    else:
        draw_boxes(frame, riders, color=(255, 200, 0), label="Rider")

    draw_boxes(frame, motorcycles, color=(0, 255, 100),  label="Motorcycle")
    draw_boxes(frame, cars,        color=(0, 180, 255),  label="Car")

    if plate_box:
        draw_boxes(frame, [plate_box], color=(0, 255, 255), label="Plate")
        if plate_text:
            x1, y1, _, _ = plate_box
            cv2.putText(frame, f"Plate: {plate_text}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # --- Violation Overlay ---
    draw_violation_overlay(frame, helmet_violation, triple_violation,
                           overspeed_violation, plate_text)

    return frame


# ---------------------------------------------------------------------------
# Video Source & Flask Streaming
# ---------------------------------------------------------------------------

def get_video_source():
    """
    Returns a cv2.VideoCapture object.
    Change argument to a video file path for offline testing, e.g. "test.mp4".
    """
    return cv2.VideoCapture(0)


def generate_frames():
    cap = get_video_source()
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = process_frame(frame)

            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        cap.release()


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
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
