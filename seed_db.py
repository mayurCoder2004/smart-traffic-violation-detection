"""
Seed PostgreSQL from users.json
================================
Bulk-inserts all vehicle records into the PostgreSQL `users` table using
ON CONFLICT DO NOTHING — safe to run multiple times, never creates duplicates.

Usage
-----
    python seed_db.py              # insert new records only (idempotent)
    python seed_db.py --reset      # DROP + recreate all tables, then seed
    python seed_db.py --check      # just print DB stats, no writes

Requirements
------------
PostgreSQL must be running.  DATABASE_URL must be set in .env.
Run  python generate_mock_data.py  first if users.json does not exist.
"""

import json
import os
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path

# ── Bootstrap Flask so SQLAlchemy is wired up ────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FLASK_ENV", "development")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from app import app, _DEMO_USERS                            # noqa: E402
from backend.extensions import db                            # noqa: E402
from backend.models import User                              # noqa: E402
from sqlalchemy import text                                  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_phone() -> str:
    return str(random.randint(6, 9)) + "".join(str(random.randint(0, 9)) for _ in range(9))


def _load_json() -> dict:
    path = ROOT / "users.json"
    records = dict(_DEMO_USERS)
    if not path.exists():
        print("[seed] ERROR: users.json not found.")
        print("       Run:  python generate_mock_data.py")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records.update(data)
    print(f"[seed] Loaded {len(data)} records from users.json (+ {len(_DEMO_USERS)} demo plates)")
    return records


def ensure_scanner_columns(ctx) -> None:
    """
    db.create_all() does not alter an existing table. Older local databases
    may have only id/name/phone/license_plate/created_at, so add the scanner
    document columns before seeding.
    """
    statements = [
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS vehicle VARCHAR(20) DEFAULT 'Unknown'",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS rc BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS insurance BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS puc BOOLEAN NOT NULL DEFAULT TRUE",
    ]
    for stmt in statements:
        ctx.session.execute(text(stmt))
    ctx.session.commit()


# ── Core seeder ───────────────────────────────────────────────────────────────

def check(session):
    """Print current DB stats without writing anything."""
    total    = session.query(User).count()
    no_ins   = session.query(User).filter_by(insurance=False).count()
    no_puc   = session.query(User).filter_by(puc=False).count()
    no_rc    = session.query(User).filter_by(rc=False).count()
    print(f"\n── DB Status {'─'*40}")
    print(f"  Total records : {total}")
    print(f"  No Insurance  : {no_ins}")
    print(f"  No PUC        : {no_puc}")
    print(f"  No RC         : {no_rc}")

    if total > 0:
        sample = session.query(User).limit(3).all()
        print(f"\n  Sample records:")
        for u in sample:
            print(f"    {u.license_plate}  |  {u.name}  |  {u.vehicle}  "
                  f"|  rc={u.rc} ins={u.insurance} puc={u.puc}")
    print(f"{'─'*52}\n")


def seed(reset: bool = False, batch_size: int = 200) -> None:
    data = _load_json()

    with app.app_context():

        # ── Optional reset ────────────────────────────────────────────────────
        if reset:
            print("[seed] Dropping all tables…")
            db.drop_all()
            print("[seed] Tables dropped.")

        # ── Ensure schema is up-to-date ───────────────────────────────────────
        print("[seed] Running db.create_all()…")
        db.create_all()
        ensure_scanner_columns(db)
        print("[seed] Schema ready.")

        # ── Build rows ────────────────────────────────────────────────────────
        now  = datetime.utcnow()
        rows = [
            {
                "id":            str(uuid.uuid4()),
                "name":          info["owner"],
                "phone":         _random_phone(),
                "license_plate": plate,
                "vehicle":       info.get("vehicle", "Car"),
                "rc":            bool(info.get("rc",        True)),
                "insurance":     bool(info.get("insurance", True)),
                "puc":           bool(info.get("puc",       True)),
                "created_at":    now,
            }
            for plate, info in data.items()
        ]

        # ── Bulk insert with ON CONFLICT DO NOTHING ───────────────────────────
        total_inserted = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            stmt  = (
                pg_insert(User.__table__)
                .values(batch)
                .on_conflict_do_nothing(index_elements=["license_plate"])
            )
            result = db.session.execute(stmt)
            db.session.commit()
            total_inserted += result.rowcount
            done = min(i + batch_size, len(rows))
            print(f"[seed]   {done}/{len(rows)} rows processed  "
                  f"(+{result.rowcount} inserted this batch)")

        total_in_db = db.session.query(User).count()
        skipped     = len(rows) - total_inserted

        print(f"\n── Seed complete {'─'*36}")
        print(f"  From JSON    : {len(rows)}")
        print(f"  Inserted     : {total_inserted}")
        print(f"  Skipped (dup): {skipped}")
        print(f"  Total in DB  : {total_in_db}")
        print(f"{'─'*52}")

        check(db.session)


def run_check() -> None:
    with app.app_context():
        check(db.session)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = set(sys.argv[1:])

    if "--check" in args:
        run_check()
        sys.exit(0)

    do_reset = "--reset" in args
    if do_reset:
        ans = input("[seed] WARNING: --reset deletes ALL data. Type 'yes' to confirm: ")
        if ans.strip().lower() != "yes":
            print("[seed] Aborted.")
            sys.exit(0)

    seed(reset=do_reset)
