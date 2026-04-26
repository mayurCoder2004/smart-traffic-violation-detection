# Smart Traffic Violation Detection System

A real-time traffic violation detection system built with Python, YOLOv8, OpenCV, and Flask. Designed for a 3-developer team with a clean modular architecture so each developer can work independently and integrate seamlessly.

---

## Team Responsibilities

| Developer | Module | Functions |
|-----------|--------|-----------|
| Developer 1 | Helmet Detection + Number Plate OCR | `detect_helmet()`, `detect_number_plate()` |
| Developer 2 | Triple Riding Detection | `detect_triple_riding()` |
| Developer 3 | Overspeed Detection | `detect_overspeed()` |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| YOLOv8 (`yolov8n.pt`) | Object detection (person, motorcycle, car) |
| OpenCV | Frame capture and drawing |
| Flask | Web server + live video streaming |
| EasyOCR / pytesseract | *(Dev 1)* Number plate text extraction |

---

## Project Structure

```
smart-traffic-violation-detection/
├── app.py                  # Main pipeline — all modules integrated here
├── requirements.txt        # Python dependencies
├── README.md
└── templates/
    └── index.html          # Live stream frontend (Flask)
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/smart-traffic-violation-detection.git
cd smart-traffic-violation-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> YOLOv8 will automatically download `yolov8n.pt` on first run.

### 4. Run the app

```bash
python app.py
```

Open your browser at `http://localhost:5000`

---

## Pipeline Overview

```
Webcam / Video File
        │
        ▼
 run_yolo_detection()
        │
        ├──► persons       (list of bounding boxes)
        ├──► motorcycles   (list of bounding boxes)
        └──► vehicles      (motorcycles + cars)
        │
        ├──► detect_helmet(frame, persons)            → helmet_violation
        ├──► detect_triple_riding(frame, persons, motorcycles) → triple_violation
        └──► detect_overspeed(frame, vehicles)        → overspeed_violation
        │
        violation = helmet_violation OR triple_violation OR overspeed_violation
        │
        if violation:
            detect_number_plate(frame)  →  plate_text, plate_box
        │
        Draw boxes + overlay text on frame
        │
        Flask /video  →  Browser
```

---

## Module Guide for Developers

### Developer 1 — Helmet Detection + Number Plate

**File:** `app.py`
**Functions to implement:** `detect_helmet()` and `detect_number_plate()`

```python
def detect_helmet(frame, persons):
    # persons: list of (x1, y1, x2, y2) bounding boxes
    # Return True if any person is detected WITHOUT a helmet
    pass

def detect_number_plate(frame):
    # Return (plate_text, plate_box) or (None, None) if not found
    pass
```

**Suggested approach:**
- Crop each person ROI from the frame using the bounding box
- Run a helmet classifier (secondary YOLO or CNN) on each ROI
- For plates: use a plate-detection model or OpenCV contour methods, then run EasyOCR on the cropped plate region

**Extra dependency to add:**
```
easyocr
```

---

### Developer 2 — Triple Riding Detection

**File:** `app.py`
**Function to implement:** `detect_triple_riding()`

```python
def detect_triple_riding(frame, persons, motorcycles):
    # persons: list of (x1, y1, x2, y2)
    # motorcycles: list of (x1, y1, x2, y2)
    # Return True if 3 or more persons are detected on a single motorcycle
    pass
```

**Suggested approach:**
- For each motorcycle bounding box, count how many person boxes overlap with it (using IoU or containment check)
- If any motorcycle has 3 or more overlapping persons → return `True`

**Overlap helper snippet:**
```python
def boxes_overlap(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1
```

---

### Developer 3 — Overspeed Detection

**File:** `app.py`
**Function to implement:** `detect_overspeed()`

```python
def detect_overspeed(frame, vehicles):
    # vehicles: list of (x1, y1, x2, y2)
    # Return True if any vehicle is estimated to be over the speed limit
    pass
```

**Suggested approach:**
- Track vehicle centroids across frames using a simple dictionary keyed by vehicle ID
- Calculate pixel displacement per frame
- Convert to km/h using a known scale factor (pixels per meter)
- Set a threshold (e.g. 60 km/h) and return `True` if any vehicle exceeds it

---

## Function Contracts (Quick Reference)

| Function | Input | Output |
|----------|-------|--------|
| `detect_helmet(frame, persons)` | BGR frame, list of person boxes | `bool` |
| `detect_number_plate(frame)` | BGR frame | `(str \| None, tuple \| None)` |
| `detect_triple_riding(frame, persons, motorcycles)` | BGR frame, person boxes, motorcycle boxes | `bool` |
| `detect_overspeed(frame, vehicles)` | BGR frame, vehicle boxes | `bool` |

> **Rule:** Each developer only modifies their own function(s). The `process_frame()` loop and all other code remains unchanged.

---

## Testing with a Video File

To test without a live webcam, edit `get_video_source()` in `app.py`:

```python
def get_video_source():
    return cv2.VideoCapture("test.mp4")   # replace with your video path
```

---

## Violation Logic & Fines

| Violation | Display Message | Fine |
|-----------|----------------|------|
| No Helmet | `No Helmet X` | ₹500 |
| Triple Riding | `Triple Riding !!!` | ₹1000 |
| Overspeed | `Overspeed >>>` | ₹1000 |

Number plate is detected and displayed **only when a violation is triggered** — avoiding unnecessary processing on clean frames.

---

## Git Workflow (Recommended)

```
main
 ├── feature/helmet-detection     ← Developer 1
 ├── feature/triple-riding        ← Developer 2
 └── feature/overspeed            ← Developer 3
```

Each developer works on their own branch and submits a PR. Since all functions have isolated signatures, merges should be conflict-free.

---

## Flask Endpoints

| Route | Description |
|-------|-------------|
| `GET /` | Web UI with live stream |
| `GET /video` | MJPEG stream (used by the `<img>` tag in the UI) |

---

## Requirements

```
flask>=3.0
opencv-python>=4.9
ultralytics>=8.0
numpy>=1.24
```

---

## License

MIT — built for hackathon use.
