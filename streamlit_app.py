"""
Triple Riding Detection — Streamlit UI
========================================
Developer 2 Module

Run with:
    streamlit run streamlit_app.py
"""

import os
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

YOLO_IMGSZ   = 320   # smaller input = faster inference
YOLO_EVERY_N = 3     # run YOLO once every N frames; draw cached boxes on the rest
DISPLAY_W    = 800   # resize frame width before sending to Streamlit

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

@st.cache_resource
def open_capture(source):
    return cv2.VideoCapture(source)

# ---------------------------------------------------------------------------
# Detection (runs every N frames) and drawing (runs every frame)
# ---------------------------------------------------------------------------

def detect(frame, model, conf_thresh):
    """Run YOLO on a resized copy and return raw bounding boxes."""
    results = model(frame, verbose=False, imgsz=YOLO_IMGSZ)[0]
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


def draw(frame, persons, motorcycles):
    """Annotate frame with cached detection boxes. Returns frame + stats."""
    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)
        cv2.putText(frame, "Person", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PERSON, 2)

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

# ---------------------------------------------------------------------------
# Detection loop — while loop means NO st.rerun(), NO flickering
# ---------------------------------------------------------------------------
if st.session_state.running and st.session_state.source is not None:
    model = load_model()
    cap   = open_capture(st.session_state.source)

    if not cap.isOpened():
        st.error("Video source closed unexpectedly.")
        st.session_state.running = False
    else:
        # This loop runs until Stop is clicked (stop_event.set()) or stream ends.
        # No st.rerun() → no page flicker. The placeholders update in-place.
        while not st.session_state.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

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
