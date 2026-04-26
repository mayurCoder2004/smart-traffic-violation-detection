"""
Smart Traffic Violation Detection System
=========================================
Team: 3 Developers
- Developer 1: Helmet Detection + Number Plate Detection + OCR
- Developer 2: Triple Riding Detection
- Developer 3: Overspeed Detection
"""

import time
import threading
import cv2
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
TRIPLE_THRESHOLD = 3

COLOR_PERSON      = (255, 200,   0)
COLOR_MOTO_NORMAL = (  0, 255, 100)
COLOR_MOTO_ALERT  = (  0,   0, 255)

YOLO_IMGSZ = 256
YOLO_EVERY_N = 8
PROCESS_WIDTH = 426
JPEG_QUALITY = 50
TARGET_FPS = 30
DETECTION_INTERVAL = 0.25
YOLO_CLASSES = [CLASS_PERSON, CLASS_MOTORCYCLE]

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
    Run YOLOv8 on a frame and extract persons and motorcycles.

    Returns:
        persons     (list[tuple]): [(x1,y1,x2,y2), ...]
        motorcycles (list[tuple]): [(x1,y1,x2,y2), ...]
    """
    results = model(
        frame,
        verbose=False,
        imgsz=YOLO_IMGSZ,
        classes=YOLO_CLASSES,
        conf=0.4,
    )[0]

    persons, motorcycles = [], []

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

    return persons, motorcycles


def is_person_on_motorcycle(person_box, moto_box, vertical_margin=0.25):
    px1, py1, px2, py2 = person_box
    mx1, my1, mx2, my2 = moto_box

    p_w = px2 - px1
    p_h = py2 - py1
    m_w = mx2 - mx1

    if p_w >= p_h:
        return False
    if p_w > m_w * 0.80:
        return False
    if not (px1 < mx2 and px2 > mx1):
        return False

    moto_height = my2 - my1
    upper_limit = my1 - vertical_margin * moto_height
    return py1 <= my2 and py2 >= upper_limit


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

def resize_for_processing(frame, target_width=PROCESS_WIDTH):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def process_frame(frame, state=None):
    """
    Streamlit-style triple-riding pipeline for compatibility with non-async use.
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
        persons, motorcycles = run_yolo_detection(frame)
        state["persons"] = persons
        state["motorcycles"] = motorcycles
    else:
        persons = state.get("persons", [])
        motorcycles = state.get("motorcycles", [])

    return draw_streamlit_logic(frame, persons, motorcycles)


# ---------------------------------------------------------------------------
# Video Source & Flask Streaming
# ---------------------------------------------------------------------------

def get_video_source():
    """
    Returns a cv2.VideoCapture object.
    Change argument to a video file path for offline testing, e.g. "test.mp4".
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    return cap


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
    def __init__(self):
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

            persons, motorcycles = run_yolo_detection(frame)
            with self.lock:
                self.persons = persons
                self.motorcycles = motorcycles

            time.sleep(DETECTION_INTERVAL)

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1)


def draw_streamlit_logic(frame, persons, motorcycles):
    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)
        cv2.putText(frame, "Person", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PERSON, 2)

    # Fallback: YOLO sometimes misses the motorcycle at lower confidence.
    # If 3+ persons are detected but no motorcycle, synthesize a motorcycle
    # bounding box from the person cluster so triple-riding can still trigger.
    if not motorcycles and len(persons) >= TRIPLE_THRESHOLD:
        sx1 = min(p[0] for p in persons)
        sy1 = min(p[1] for p in persons)
        sx2 = max(p[2] for p in persons)
        sy2 = max(p[3] for p in persons)
        motorcycles = [(sx1, sy1, sx2, sy2)]

    any_triple = False

    for (mx1, my1, mx2, my2) in motorcycles:
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
                        (mx1, my1 - 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_MOTO_ALERT, 2)
        else:
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), COLOR_MOTO_NORMAL, 2)
            cv2.putText(frame, f"Motorcycle  Riders: {rider_count}",
                        (mx1, my1 - 6),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_MOTO_NORMAL, 2)

    if any_triple:
        cv2.rectangle(frame, (0, 0), (360, 60), (0, 0, 200), -1)
        cv2.putText(frame, "TRIPLE RIDING DETECTED", (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(frame, "Fine: Rs.1000", (8, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

    return frame


def generate_frames():
    raw_cap = get_video_source()
    if not raw_cap.isOpened():
        raise RuntimeError("Cannot open video source.")

    cap = LatestFrameCapture(raw_cap)
    detector = AsyncYoloDetector()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                time.sleep(0.01)
                continue

            frame = resize_for_processing(frame)
            detector.submit(frame)
            persons, motorcycles = detector.get_result()
            frame = draw_streamlit_logic(frame, persons, motorcycles)

            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ret:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        detector.release()
        cap.release()


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
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
