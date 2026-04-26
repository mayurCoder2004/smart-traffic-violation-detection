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
import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.40          # minimum detection confidence
TRIPLE_RIDING_THRESHOLD = 3    # persons per motorcycle to trigger alert

# COCO class IDs
CLASS_PERSON     = 0
CLASS_MOTORCYCLE = 3

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

def process_frame(frame, model):
    """
    Run YOLO, evaluate triple-riding for each motorcycle, draw results.

    Returns the annotated frame.
    """
    results = model(frame, verbose=False)[0]

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


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0  # 0 = default webcam

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"[INFO] Opening video source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("[ERROR] Cannot open video source.")
        sys.exit(1)

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        frame = process_frame(frame, model)
        cv2.imshow("Triple Riding Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
