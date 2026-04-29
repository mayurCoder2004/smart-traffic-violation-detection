# Smart Traffic Violation Detection

Real-time traffic violation detection and challan management system built with Flask, YOLOv8, OpenCV, PostgreSQL, Razorpay, and a React dashboard.

The app detects traffic violations from webcam/video feeds, records violations and scanner challans, lets police issue challans manually, and lets users view and pay challans for their vehicle.

## Features

- Live violation detection stream for helmet, triple riding, and overspeed checks.
- Smart police scanner for vehicle lookup, document verification, and challan issuing.
- User challan dashboard with Razorpay checkout for payments.
- Police dashboard for viewing all detected violations.
- Traffic signal simulation/monitoring with its own video feed and status API.
- PostgreSQL-backed users, violations, scanner challans, challan items, and payments.
- `users.json` mock vehicle dataset with seeding support.
- 24-hour duplicate rule: one challan per vehicle per violation type within 24 hours.
- IST timestamps in API responses.

## Tech Stack

| Layer | Tools |
| --- | --- |
| Backend | Python, Flask, Flask-CORS, Flask-SQLAlchemy |
| CV/ML | OpenCV, Ultralytics YOLOv8, EasyOCR, Pillow |
| Frontend | React, Vite, Tailwind CSS, Framer Motion, Lucide React |
| Database | PostgreSQL, psycopg2 |
| Payments | Razorpay |
| Utilities | Docker Compose, python-dotenv |

## Project Structure

```text
smart-traffic-violation-detection/
├── app.py                         # Main Flask app, CV streams, scanner APIs, payment APIs
├── helmet_detector.py             # Helmet detector wrapper
├── overspeed_detector.py          # Overspeed logic/helpers
├── triple_riding.py               # Triple-riding detection module
├── traffic_signal.py              # Traffic signal processing system
├── streamlit_app.py               # Optional Streamlit entry point
├── generate_mock_data.py          # Generates users.json vehicle records
├── seed_db.py                     # Seeds PostgreSQL from users.json
├── migrate_docker_to_neon.py      # Helper for migrating local DB data to Neon
├── users.json                     # Mock vehicle registry dataset
├── requirements.txt               # Python dependencies
├── docker-compose.yml             # Local PostgreSQL service
├── .env.example                   # Environment variable template
├── backend/
│   ├── config.py                  # Flask config
│   ├── extensions.py              # SQLAlchemy instance
│   ├── models.py                  # User, Violation, ScannerChallan, Payment models
│   └── routes/
│       ├── users.py               # /users APIs
│       ├── violations.py          # /violations APIs
│       └── payments.py            # Razorpay APIs for legacy violations
├── frontend/
│   ├── package.json               # React app dependencies/scripts
│   ├── vite.config.js             # Vite dev server and proxy config
│   └── src/
│       ├── App.jsx                # React routes
│       ├── api/index.js           # Axios helpers
│       └── components/
│           ├── Login.jsx
│           ├── ChallanUser.jsx
│           ├── ChallanPolice.jsx
│           ├── PoliceDashboard.jsx
│           ├── TrafficSignal.jsx
│           ├── PaymentButton.jsx
│           └── Navbar.jsx
└── templates/
    └── index.html                 # Basic Flask stream page
```

## Prerequisites

- Python 3.9+.
- Node.js 18+.
- PostgreSQL, either local via Docker Compose or hosted, such as Neon.
- Razorpay test keys if payment windows should open.
- Model files where used by the app:
  - `yolov8n.pt` is downloaded by Ultralytics if missing.
  - `helmet_model.pt` is expected locally for helmet detection.
  - `license_plate_detector.pt` is expected locally for plate detection.

## Environment

Create `.env` from the example:

```bash
cp .env.example .env
```

Required variables:

```env
DATABASE_URL=postgresql://traffic_user:traffic_pass@localhost:5432/traffic_violations
RAZORPAY_KEY_ID=your_razorpay_key_id_here
RAZORPAY_KEY_SECRET=your_razorpay_key_secret_here
SECRET_KEY=change-this-to-a-random-secret-key
FLASK_ENV=development
```

Do not commit real `.env` secrets.

## Backend Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Start local PostgreSQL if you are not using a hosted database:

```bash
docker compose up -d db
```

Generate mock vehicle data if needed:

```bash
python generate_mock_data.py
```

Seed the database:

```bash
python seed_db.py
```

Useful seeding commands:

```bash
python seed_db.py --check
python seed_db.py --reset
```

Start Flask:

```bash
python app.py
```

The backend runs on:

```text
http://localhost:9000
```

## Frontend Setup

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

Build production assets:

```bash
cd frontend
npm run build
```

## Main App Pages

| Page | Route | Purpose |
| --- | --- | --- |
| Login | `/login` | Vehicle plate login |
| My Challans | `/dashboard` | User challan list and Razorpay payment |
| Police Dashboard | `/police` | All detected violations |
| Traffic Signal | `/signal` | Signal video/status UI |
| Police Scanner | `/scanner` | Scan plate, verify documents, issue challans |

## Backend APIs

### Core Video and Detection

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/` | Basic Flask stream page |
| `GET` | `/video` | Main MJPEG violation detection stream |
| `POST` | `/upload` | Upload a video for detection |
| `POST` | `/use_webcam` | Switch main detection source to webcam |
| `GET` | `/source_status` | Current video source metadata |
| `GET` | `/detections` | Latest detection status |

### Traffic Signal

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/signal_video` | Traffic signal MJPEG stream |
| `GET` | `/signal_status` | Signal counts, current green lane, countdown |
| `POST` | `/signal_use_webcam` | Switch signal source to webcam |
| `POST` | `/signal_upload` | Upload a signal video |

### Users and Violations

| Method | Route | Description |
| --- | --- | --- |
| `POST` | `/users` | Create a user |
| `POST` | `/users/login` | Plate-based login |
| `GET` | `/users` | List users |
| `GET` | `/users/<plate>` | Get user by plate |
| `POST` | `/violations` | Create a legacy violation |
| `GET` | `/violations` | List all violations |
| `GET` | `/violations/user/<user_id>` | List violations for one user |
| `GET` | `/violations/<violation_id>` | Get one violation |

### Scanner and Challans

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/scan/<plate>` | Lookup vehicle from `users.json`, then PostgreSQL fallback |
| `GET` | `/user/<plate>` | User profile plus scanner challans |
| `POST` | `/create-challan` | Issue scanner challan |
| `POST` | `/pay/<challan_id>` | Mark challan paid, kept for fallback/manual flows |
| `GET` | `/all-challans` | List scanner challans |
| `GET` | `/sample-plates` | Get sample plates for scanner quick-fill |

### Payments

| Method | Route | Description |
| --- | --- | --- |
| `POST` | `/payments/create-order` | Create Razorpay order for legacy violations |
| `POST` | `/payments/verify` | Verify Razorpay payment for legacy violations |
| `POST` | `/payments/create-scanner-order` | Create Razorpay order for scanner challans |
| `POST` | `/payments/verify-scanner` | Verify scanner challan payment |

## Data Model Summary

- `User`: owner, phone, license plate, vehicle type, RC/insurance/PUC status.
- `Violation`: CV/legacy violation row with status and fine.
- `ScannerChallan`: challan issued from scanner or CV pipeline.
- `ScannerChallanItem`: individual violation/document item inside a challan.
- `Payment`: Razorpay order/payment record for legacy violations.

## Violation Types and Fines

| Violation | Fine |
| --- | ---: |
| No Helmet | Rs. 500 |
| Triple Riding | Rs. 1000 |
| Overspeed | Rs. 1000 |
| No Insurance | Rs. 1000 |
| No PUC | Rs. 500 |

Scanner challans enforce one challan item per vehicle per violation type in a rolling 24-hour window.

## Common Workflows

### Register and seed vehicles

```bash
python generate_mock_data.py 1000
python seed_db.py
```

### Check if DB has records

```bash
python seed_db.py --check
```

### Run the full app locally

Terminal 1:

```bash
source venv/bin/activate
python app.py
```

Terminal 2:

```bash
cd frontend
npm run dev
```

### Use Police Scanner

1. Open `http://localhost:5173/scanner`.
2. Enter or quick-fill a plate number.
3. Scan/verify the registered owner and documents.
4. Select violation types.
5. Issue challan.
6. User can view/pay it from `/dashboard`.

## Notes

- The frontend is configured to talk to `http://localhost:9000`.
- Some frontend API helpers use Vite proxies, while scanner pages call the backend directly.
- If backend code changes, restart `python app.py`.
- If the payment window does not open, confirm Razorpay keys are present and the checkout script is loaded from `frontend/index.html`.
- If a DB-only plate scans correctly but challan issuing fails, restart Flask so the latest backend lookup code is loaded.

## Troubleshooting

### `NotOpenSSLWarning` on macOS

This warning comes from urllib3 with Python builds linked against LibreSSL. The app can still run. Using a Python.org/Homebrew Python linked with OpenSSL removes the warning.

### Camera is busy or blank

Close other apps using the webcam, then restart Flask. You can also upload a video from the UI.

### Tables missing columns

Run:

```bash
python seed_db.py
```

The seeder applies missing scanner columns before inserting rows.

### Duplicate challan skipped

This is intentional. The scanner blocks duplicate challan items for the same vehicle and violation type within 24 hours.

## License

MIT.
