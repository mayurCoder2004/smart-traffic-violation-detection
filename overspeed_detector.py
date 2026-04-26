"""
Overspeed Detection — Developer 3
==================================
Single-vehicle real-time speed estimator using two virtual trip lines.

Algorithm
---------
  Speed = REAL_DISTANCE / time_taken
  where time_taken = time at LINE2 - time at LINE1

Single-vehicle guarantee
------------------------
  Phase 1 — IDLE    : no vehicle locked, waiting for first detection
  Phase 2 — LOCKED  : centroid-tracked across frames, waiting for LINE1
  Phase 3 — TIMING  : vehicle crossed LINE1, timer running
  Phase 4 — RESULT  : vehicle crossed LINE2, speed frozen on screen
  Reset              : vehicle lost for MISSING_FRAMES consecutive frames

Controls
--------
  Q  — quit
  R  — reset and re-lock on the next vehicle
"""

import time
import math
import sys
import os
import cv2
from ultralytics import YOLO


# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_PATH        = "yolov8n.pt"

LINE1_Y           = 160     # upper trip line (pixels) — starts timer
LINE2_Y           = 420     # lower trip line (pixels) — stops timer
REAL_DISTANCE     = 10.0    # real-world gap between lines (metres)
SPEED_LIMIT       = 40.0    # km/h — flag anything above this

CONF_THRESHOLD    = 0.40    # minimum YOLO detection confidence
MISSING_FRAMES    = 15      # consecutive missed frames before auto-reset
MAX_CENTROID_DIST = 150     # max pixel jump still considered the same vehicle

# COCO class IDs for road vehicles
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Bounding-box colour by phase
COLOR_LOCKED    = (0,   255, 100)   # green  — tracking, not yet at LINE1
COLOR_TIMING    = (0,   165, 255)   # orange — between the two lines
COLOR_SAFE      = (0,   220, 0  )   # bright green — finished, within limit
COLOR_OVERSPEED = (0,   0,   255)   # red    — finished, over limit


# ── Model ──────────────────────────────────────────────────────────────────────

model = YOLO(MODEL_PATH)


# ── State ──────────────────────────────────────────────────────────────────────

class SpeedState:
    """
    All mutable state for one measurement cycle.

    Phases
    ------
    IDLE    → locked_on=False
    LOCKED  → locked_on=True,  crossed_line1=False
    TIMING  → locked_on=True,  crossed_line1=True,  crossed_line2=False
    RESULT  → locked_on=True,  crossed_line2=True,  speed_kmph set
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # tracking
        self.locked_on     = False
        self.last_cx       = None   # centroid x of the tracked vehicle last frame
        self.last_cy       = None   # centroid y of the tracked vehicle last frame
        self.label         = ""     # class label of the locked vehicle

        # timing
        self.start_time    = None
        self.crossed_line1 = False
        self.crossed_line2 = False

        # result (frozen after LINE2)
        self.speed_kmph    = None
        self.overspeeding  = False

        # housekeeping
        self.missing_count = 0

    @property
    def phase(self):
        if not self.locked_on:
            return "IDLE"
        if not self.crossed_line1:
            return "LOCKED"
        if not self.crossed_line2:
            return "TIMING"
        return "RESULT"


# ── Detection & Tracking ───────────────────────────────────────────────────────

def get_all_vehicles(frame):
    """
    Run YOLOv8 on frame and return every vehicle detection.

    Returns:
        list of (x1, y1, x2, y2, label, cx, cy)
    """
    results = model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        if cls in VEHICLE_CLASSES and conf >= CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append((x1, y1, x2, y2, VEHICLE_CLASSES[cls], cx, cy))
    return detections


def find_tracked_vehicle(state, detections):
    """
    Return the detection that corresponds to the currently tracked vehicle.

    Lock-on  : if no vehicle is locked yet, return the first detection.
    Tracking : return the detection whose centroid is closest to the last
               known centroid, provided it is within MAX_CENTROID_DIST.
               Return None if nothing qualifies.

    Args:
        state      (SpeedState): current state (reads last_cx / last_cy)
        detections (list)      : output of get_all_vehicles()

    Returns:
        One detection tuple  or  None
    """
    if not detections:
        return None

    # Phase IDLE — lock onto the very first detection seen
    if not state.locked_on:
        return detections[0]

    # Phases LOCKED / TIMING / RESULT — match by nearest centroid
    best     = None
    best_d   = float("inf")
    for det in detections:
        *_, cx, cy = det
        d = math.hypot(cx - state.last_cx, cy - state.last_cy)
        if d < best_d:
            best_d = d
            best   = det

    # Reject if the closest candidate jumped too far (different vehicle)
    if best_d > MAX_CENTROID_DIST:
        return None

    return best


# ── Timing ─────────────────────────────────────────────────────────────────────

def update_timing(state, cy):
    """
    Check centroid y against trip lines and advance the state machine.
    Speed is computed exactly once; the RESULT phase is never re-entered.
    """
    if state.crossed_line2:
        return  # already in RESULT — nothing to do

    if not state.crossed_line1 and cy >= LINE1_Y:
        state.start_time    = time.time()
        state.crossed_line1 = True
        print("[TIMER] Started timing.")

    elif state.crossed_line1 and cy >= LINE2_Y:
        elapsed             = time.time() - state.start_time
        speed_mps           = REAL_DISTANCE / elapsed
        state.speed_kmph    = round(speed_mps * 3.6, 2)
        state.overspeeding  = state.speed_kmph > SPEED_LIMIT
        state.crossed_line2 = True

        print(f"[TIMER] Stopped timing.  Elapsed: {elapsed:.3f} s")
        print(f"[SPEED] {state.speed_kmph} km/h  —  "
              f"{'*** OVERSPEEDING ***' if state.overspeeding else 'OK'}")


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_trip_lines(frame):
    """Draw the two horizontal trip lines."""
    w = frame.shape[1]

    cv2.line(frame, (0, LINE1_Y), (w, LINE1_Y), (0, 200, 255), 2)
    cv2.putText(frame, "LINE 1 — Start Timer", (10, LINE1_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

    cv2.line(frame, (0, LINE2_Y), (w, LINE2_Y), (255, 200, 0), 2)
    cv2.putText(frame, "LINE 2 — Stop Timer", (10, LINE2_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)


def pick_box_color(state):
    """Return the bounding-box colour appropriate for the current phase."""
    if state.phase == "RESULT":
        return COLOR_OVERSPEED if state.overspeeding else COLOR_SAFE
    if state.phase == "TIMING":
        return COLOR_TIMING
    return COLOR_LOCKED


def draw_vehicle(frame, x1, y1, x2, y2, label, cx, cy, state):
    """Draw bounding box, label, and centroid for the locked vehicle."""
    color = pick_box_color(state)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{label}  [LOCKED]", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)      # centroid dot
    cv2.circle(frame, (cx, cy), 6, (0, 0, 0),     1)       # outline for contrast


def draw_hud(frame, state):
    """Render the speed result or live status in the top-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h    = frame.shape[0]

    if state.phase == "RESULT":
        color = COLOR_OVERSPEED if state.overspeeding else COLOR_SAFE
        cv2.putText(frame, f"Speed : {state.speed_kmph:.1f} km/h",
                    (10, 38), font, 0.8, color, 2)
        cv2.putText(frame, f"Limit : {SPEED_LIMIT:.0f} km/h",
                    (10, 68), font, 0.65, (180, 180, 180), 1)
        if state.overspeeding:
            cv2.putText(frame, "OVERSPEEDING!", (10, 105), font, 1.1,
                        COLOR_OVERSPEED, 3)
        else:
            cv2.putText(frame, "Within Speed Limit", (10, 105), font, 0.8,
                        COLOR_SAFE, 2)

    elif state.phase == "TIMING":
        elapsed = time.time() - state.start_time
        cv2.putText(frame, f"Timing...  {elapsed:.2f} s", (10, 38),
                    font, 0.75, (255, 165, 0), 2)
        cv2.putText(frame, "Waiting for LINE 2", (10, 68),
                    font, 0.55, (200, 200, 200), 1)

    elif state.phase == "LOCKED":
        cv2.putText(frame, f"Tracking: {state.label}", (10, 38),
                    font, 0.7, COLOR_LOCKED, 2)
        cv2.putText(frame, "Waiting for LINE 1", (10, 68),
                    font, 0.55, (200, 200, 200), 1)

    else:  # IDLE
        cv2.putText(frame, "Waiting for a vehicle...", (10, 38),
                    font, 0.65, (150, 150, 150), 1)

    # footer
    cv2.putText(frame, "Q: quit   R: reset", (10, h - 12),
                font, 0.45, (80, 80, 80), 1)


# ── Main Loop ──────────────────────────────────────────────────────────────────

def main(video_source=0):
    """
    Run the overspeed detector on a video source.
    
    Args:
        video_source: Either an integer (0 for webcam) or a string (path to video file)
    """
    # If a file path is provided, validate it exists
    if isinstance(video_source, str):
        if not os.path.exists(video_source):
            print(f"[ERROR] Video file not found: {video_source}")
            print(f"[ERROR] Current working directory: {os.getcwd()}")
            print(f"[ERROR] Please provide a valid video file path.")
            return
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        source_name = "webcam (device 0)" if video_source == 0 else f"video file '{video_source}'"
        raise RuntimeError(f"Cannot open {source_name}.")

    state = SpeedState()

    source_name = "Webcam (device 0)" if video_source == 0 else f"Video file: {video_source}"
    print("[INFO] Overspeed detector started.")
    print(f"[INFO] Source: {source_name}")
    print(f"[INFO] LINE1 y={LINE1_Y}  |  LINE2 y={LINE2_Y}  |  "
          f"Distance={REAL_DISTANCE}m  |  Limit={SPEED_LIMIT}km/h")
    print("[INFO] Press Q to quit, R to reset.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame read failed — exiting.")
            break

        detections = get_all_vehicles(frame)
        tracked    = find_tracked_vehicle(state, detections)

        # ── Vehicle not found this frame ──────────────────────────────────────
        if tracked is None:
            state.missing_count += 1

            if state.missing_count >= MISSING_FRAMES:
                # Only log the reset if we had actually started tracking
                if state.locked_on:
                    print(f"[INFO] Vehicle lost for {MISSING_FRAMES} frames "
                          f"— resetting.\n")
                state.reset()

        # ── Vehicle found ─────────────────────────────────────────────────────
        else:
            x1, y1, x2, y2, label, cx, cy = tracked
            state.missing_count = 0

            # Lock-on: first time we see a vehicle
            if not state.locked_on:
                state.locked_on = True
                state.label     = label
                print(f"[INFO] Locked onto {label} at centroid ({cx}, {cy}).")

            # Update tracked centroid so next frame can match correctly
            state.last_cx = cx
            state.last_cy = cy

            # Advance timing state machine
            update_timing(state, cy)

            # Draw the vehicle
            draw_vehicle(frame, x1, y1, x2, y2, label, cx, cy, state)

        # ── Always-on overlay ─────────────────────────────────────────────────
        draw_trip_lines(frame)
        draw_hud(frame, state)

        cv2.imshow("Overspeed Detector — Dev 3", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            state.reset()
            print("[INFO] Manual reset — waiting for next vehicle.\n")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detector stopped.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])       # e.g. python overspeed_detector.py video.mp4
    else:
        main()                  # default: webcam (device 0)
