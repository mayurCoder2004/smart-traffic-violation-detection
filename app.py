"""
Smart Traffic Violation Detection System
=========================================
Team: 3 Developers
- Developer 1: Helmet Detection + Number Plate Detection + OCR
- Developer 2: Triple Riding Detection
- Developer 3: Overspeed Detection
"""

import cv2
import numpy as np
from flask import Flask, Response, render_template
from ultralytics import YOLO

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------------------------
model = YOLO("yolov8n.pt")

# YOLO class IDs (COCO dataset)
CLASS_PERSON     = 0
CLASS_MOTORCYCLE = 3
CLASS_CAR        = 2

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
    for (mx1, my1, mx2, my2) in motorcycles:
        moto_height = my2 - my1
        moto_width  = mx2 - mx1
        upper_limit = my1 - 0.25 * moto_height

        rider_count = 0
        for (px1, py1, px2, py2) in persons:
            p_w = px2 - px1
            p_h = py2 - py1

            # Reject vehicle-like boxes: real persons are taller than wide
            if p_w >= p_h:
                continue
            # Reject boxes much wider than the motorcycle (e.g. a passing car)
            if p_w > moto_width * 0.80:
                continue
            # Must overlap horizontally with the motorcycle
            if not (px1 < mx2 and px2 > mx1):
                continue
            # Must be positioned above or overlapping (not fully below the motorcycle)
            if py1 <= my2 and py2 >= upper_limit:
                rider_count += 1

        if rider_count >= 3:
            # Draw red box + alert text directly onto frame for this motorcycle
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 0, 255), 3)
            cv2.putText(frame, "Triple Riding!", (mx1, my1 - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.putText(frame, f"Riders: {rider_count}  |  Fine: Rs.1000",
                        (mx1, my1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            return True

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
    app.run(host="0.0.0.0", port=5000, debug=False)
