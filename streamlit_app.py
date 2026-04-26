"""
Triple Riding Detection — Streamlit UI
========================================
Developer 2 Module

Run with:
    streamlit run streamlit_app.py
"""

import os
import time
import cv2
import tempfile
import threading
import streamlit as st
from ultralytics import YOLO
from triple_riding import is_person_on_motorcycle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH       = "yolov8n.pt"
CLASS_PERSON     = 0
CLASS_MOTORCYCLE = 3
TRIPLE_THRESHOLD = 3

COLOR_PERSON      = (255, 200,   0)
COLOR_MOTO_NORMAL = (  0, 255, 100)
COLOR_MOTO_ALERT  = (  0,   0, 255)

YOLO_IMGSZ   = 256   # smaller input = faster inference
YOLO_EVERY_N = 8     # run YOLO once every N frames; draw cached boxes on the rest
DISPLAY_W    = 426   # resize frame width before inference and Streamlit display
DETECTION_INTERVAL = 0.25
YOLO_CLASSES = [CLASS_PERSON, CLASS_MOTORCYCLE]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Triple Riding Detection",
    page_icon="🚨",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resources — loaded once, never re-created on rerun
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


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


@st.cache_resource
def open_capture(source):
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if source == 0 else cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    return LatestFrameCapture(cap) if source == 0 else cap

# ---------------------------------------------------------------------------
# Detection (runs every N frames) and drawing (runs every frame)
# ---------------------------------------------------------------------------

def detect(frame, model, conf_thresh):
    """Run YOLO on a resized copy and return raw bounding boxes."""
    results = model(
        frame,
        verbose=False,
        imgsz=YOLO_IMGSZ,
        classes=YOLO_CLASSES,
        conf=conf_thresh,
    )[0]
    persons, motorcycles = [], []
    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == CLASS_PERSON:
            persons.append((x1, y1, x2, y2))
        elif cls == CLASS_MOTORCYCLE:
            motorcycles.append((x1, y1, x2, y2))
    return persons, motorcycles


class AsyncYoloDetector:
    def __init__(self, model, conf_thresh):
        self.model = model
        self.conf_thresh = conf_thresh
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

            persons, motorcycles = detect(frame, self.model, self.conf_thresh)
            with self.lock:
                self.persons = persons
                self.motorcycles = motorcycles

            time.sleep(DETECTION_INTERVAL)

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1)


def resize_frame(frame, target_width=DISPLAY_W):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def draw(frame, persons, motorcycles):
    """Annotate frame with cached detection boxes. Returns frame + stats."""
    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)
        cv2.putText(frame, "Person", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PERSON, 2)

    # Fallback: if YOLO missed the motorcycle but found 3+ persons,
    # synthesize a motorcycle box from the person cluster.
    if not motorcycles and len(persons) >= TRIPLE_THRESHOLD:
        sx1 = min(p[0] for p in persons)
        sy1 = min(p[1] for p in persons)
        sx2 = max(p[2] for p in persons)
        sy2 = max(p[3] for p in persons)
        motorcycles = [(sx1, sy1, sx2, sy2)]

    any_triple = False
    max_riders  = 0

    for (mx1, my1, mx2, my2) in motorcycles:
        rider_count = sum(
            1 for p in persons
            if is_person_on_motorcycle(p, (mx1, my1, mx2, my2))
        )
        max_riders = max(max_riders, rider_count)
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

    return frame, len(persons), len(motorcycles), any_triple, max_riders


def process_frame(frame, model, conf_thresh):
    frame = resize_frame(frame)

    if "frame_count" not in st.session_state:
        st.session_state.frame_count = 0
    if "cached_persons" not in st.session_state:
        st.session_state.cached_persons = []
    if "cached_motorcycles" not in st.session_state:
        st.session_state.cached_motorcycles = []

    st.session_state.frame_count += 1
    should_detect = (
        st.session_state.frame_count == 1
        or st.session_state.frame_count % YOLO_EVERY_N == 0
    )

    if should_detect:
        persons, motorcycles = detect(frame, model, conf_thresh)
        st.session_state.cached_persons = persons
        st.session_state.cached_motorcycles = motorcycles
    else:
        persons = st.session_state.cached_persons
        motorcycles = st.session_state.cached_motorcycles

    return draw(frame, persons, motorcycles)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "source" not in st.session_state:
    st.session_state.source = None
if "tmp_path" not in st.session_state:
    st.session_state.tmp_path = None
# threading.Event shared by reference across reruns — the loop reads the same object
# that the Stop button sets, so Stop takes effect within one frame.
if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()
    st.session_state.stop_event.set()   # starts in "stopped" state

# ---------------------------------------------------------------------------
# Sidebar — settings only
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    source_type = st.radio("Video Source", ["Webcam", "Video File"])

    uploaded_file = None
    if source_type == "Video File":
        uploaded_file = st.file_uploader(
            "Upload a video", type=["mp4", "avi", "mov", "mkv"]
        )

    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.90, 0.40, 0.05)

    st.divider()
    st.markdown("**Legend**")
    st.markdown("🟡 Person")
    st.markdown("🟢 Motorcycle — safe")
    st.markdown("🔴 Motorcycle — triple riding")

# ---------------------------------------------------------------------------
# Header + Start / Stop buttons (main area — always visible)
# ---------------------------------------------------------------------------
st.title("🚦 Smart Traffic Violation Detection")
st.caption("Triple Riding Module — YOLOv8n · Person-only association")

btn_c1, btn_c2, _ = st.columns([1, 1, 5])
start_btn = btn_c1.button(
    "▶ Start", use_container_width=True, type="primary",
    disabled=st.session_state.running,
)
stop_btn = btn_c2.button(
    "⏹ Stop", use_container_width=True,
    disabled=not st.session_state.running,
)

st.divider()

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
video_col, stats_col = st.columns([3, 1])

with video_col:
    frame_placeholder = st.empty()

with stats_col:
    st.subheader("Live Stats")
    metric_persons   = st.empty()
    metric_motos     = st.empty()
    metric_riders    = st.empty()
    metric_violation = st.empty()
    alert_box        = st.empty()

# ---------------------------------------------------------------------------
# Stop button — signal the running loop to exit
# ---------------------------------------------------------------------------
if stop_btn:
    st.session_state.stop_event.set()   # loop checks this every frame
    st.session_state.running = False
    open_capture.clear()                # release the camera / file
    if st.session_state.tmp_path:
        try:
            os.unlink(st.session_state.tmp_path)
        except OSError:
            pass
        st.session_state.tmp_path = None
    st.session_state.source = None

# ---------------------------------------------------------------------------
# Start button — open source and kick off the while loop
# ---------------------------------------------------------------------------
if start_btn:
    # Prepare source
    if source_type == "Webcam":
        open_capture.clear()            # release any previous capture first
        st.session_state.source = 0

    else:
        if uploaded_file is None:
            st.warning("Please upload a video file first.")
            st.stop()

        if st.session_state.tmp_path:
            try:
                os.unlink(st.session_state.tmp_path)
            except OSError:
                pass

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded_file.read())
        tmp.close()
        st.session_state.tmp_path = tmp.name

        open_capture.clear()
        st.session_state.source = st.session_state.tmp_path

    # Verify the source opens before entering the loop
    test_cap = cv2.VideoCapture(st.session_state.source)
    if not test_cap.isOpened():
        st.error("Cannot open video source. Check webcam permissions or re-upload.")
        test_cap.release()
        st.stop()
    test_cap.release()

    st.session_state.stop_event.clear() # clear any previous stop signal
    st.session_state.running = True
    st.session_state.frame_count = 0
    st.session_state.cached_persons = []
    st.session_state.cached_motorcycles = []

# ---------------------------------------------------------------------------
# Detection loop — while loop means NO st.rerun(), NO flickering
# ---------------------------------------------------------------------------
if st.session_state.running and st.session_state.source is not None:
    model = load_model()
    cap   = open_capture(st.session_state.source)
    detector = AsyncYoloDetector(model, conf_thresh) if st.session_state.source == 0 else None

    if not cap.isOpened():
        st.error("Video source closed unexpectedly.")
        st.session_state.running = False
    else:
        # This loop runs until Stop is clicked (stop_event.set()) or stream ends.
        # No st.rerun() → no page flicker. The placeholders update in-place.
        while not st.session_state.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if st.session_state.source == 0:
                    time.sleep(0.01)
                    continue
                break

            if detector is not None:
                frame = resize_frame(frame)
                detector.submit(frame)
                persons, motorcycles = detector.get_result()
                frame, n_p, n_m, triple, max_r = draw(frame, persons, motorcycles)
            else:
                frame, n_p, n_m, triple, max_r = process_frame(frame, model, conf_thresh)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            metric_persons.metric("Persons Detected", n_p)
            metric_motos.metric("Motorcycles", n_m)
            metric_riders.metric("Max Riders on One Moto", max_r)

            if triple:
                metric_violation.metric("Violation", "YES ⚠️")
                alert_box.error("**TRIPLE RIDING DETECTED**\n\nFine: ₹1000")
            else:
                metric_violation.metric("Violation", "None")
                alert_box.success("No violation detected")

        # Loop exited — clean up
        st.session_state.running = False
        if detector is not None:
            detector.release()
        cap.release()
        open_capture.clear()
        if st.session_state.tmp_path:
            try:
                os.unlink(st.session_state.tmp_path)
            except OSError:
                pass
            st.session_state.tmp_path = None
        st.session_state.source = None
        frame_placeholder.info("Stream ended. Press **▶ Start** to restart.")

else:
    if not st.session_state.running:
        frame_placeholder.info("Press **▶ Start** above to begin detection.")
        metric_persons.metric("Persons Detected", "--")
        metric_motos.metric("Motorcycles", "--")
        metric_riders.metric("Max Riders on One Moto", "--")
        metric_violation.metric("Violation", "--")
