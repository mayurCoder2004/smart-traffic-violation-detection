"""
Mock Vehicle Database Generator
================================
Generates n unique vehicle records for the Smart Police Scanner system
and saves them to users.json.

Usage:
    python generate_mock_data.py            # generates 1000 records
    python generate_mock_data.py 500        # generates 500 records

Importable:
    from generate_mock_data import generate_users
    users = generate_users(n=200)
"""

import json
import random
import string
import sys
from pathlib import Path

# ── Indian first + last names pool ────────────────────────────────────────────
FIRST_NAMES = [
    "Aarav",    "Arjun",    "Rohan",    "Vikram",   "Karthik",
    "Rahul",    "Amit",     "Sanjay",   "Suresh",   "Rajesh",
    "Priya",    "Sneha",    "Ananya",   "Divya",    "Kavya",
    "Neha",     "Pooja",    "Shreya",   "Meera",    "Lakshmi",
    "Raghav",   "Aditya",   "Manish",   "Deepak",   "Nikhil",
    "Ravi",     "Harish",   "Ganesh",   "Prasad",   "Venkat",
    "Sunita",   "Geeta",    "Rekha",    "Usha",     "Shanta",
    "Mohan",    "Gopal",    "Ramesh",   "Sunil",    "Vijay",
]

LAST_NAMES = [
    "Sharma",   "Verma",    "Gupta",    "Singh",    "Kumar",
    "Patel",    "Mehta",    "Joshi",    "Nair",     "Pillai",
    "Reddy",    "Rao",      "Iyer",     "Menon",    "Krishnan",
    "Mishra",   "Tiwari",   "Pandey",   "Dubey",    "Shukla",
    "Agarwal",  "Bansal",   "Malhotra", "Kapoor",   "Khanna",
    "Chatterjee","Mukherjee","Bose",    "Das",      "Roy",
    "Dhingra",  "Bhatia",   "Anand",    "Saxena",   "Srivastava",
]

VEHICLE_TYPES  = ["Bike", "Car", "Truck"]
VEHICLE_WEIGHTS = [0.45, 0.40, 0.15]   # bikes most common, trucks least


def _random_plate(existing: set) -> str:
    """Generate a unique KA-format plate not already in *existing*."""
    while True:
        rto    = str(random.randint(1, 99)).zfill(2)
        series = random.choice(string.ascii_uppercase) + random.choice(string.ascii_uppercase)
        number = str(random.randint(1, 9999)).zfill(4)
        plate  = f"KA{rto}{series}{number}"
        if plate not in existing:
            existing.add(plate)
            return plate


def _random_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def generate_users(n: int = 1000) -> dict:
    """
    Generate *n* unique vehicle records.

    Document probabilities:
        rc        — 90 % True
        insurance — 70 % True
        puc       — 70 % True

    Returns:
        dict  { plate_string: { owner, vehicle, rc, insurance, puc, challans } }
    """
    users   = {}
    seen    = set()

    for _ in range(n):
        plate = _random_plate(seen)
        users[plate] = {
            "owner":     _random_name(),
            "vehicle":   random.choices(VEHICLE_TYPES, weights=VEHICLE_WEIGHTS, k=1)[0],
            "rc":        random.random() < 0.90,
            "insurance": random.random() < 0.70,
            "puc":       random.random() < 0.70,
            "challans":  [],
        }

    return users


def save_users(users: dict, path: str = "users.json") -> None:
    """Serialise *users* to a JSON file."""
    out = Path(path)
    out.write_text(json.dumps(users, indent=2, ensure_ascii=False))
    print(f"[generate_mock_data] Saved {len(users)} records → {out.resolve()}")


# ── CLI entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n     = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    users = generate_users(n)
    save_users(users)

    # Quick statistics
    total       = len(users)
    no_ins      = sum(1 for u in users.values() if not u["insurance"])
    no_puc      = sum(1 for u in users.values() if not u["puc"])
    no_rc       = sum(1 for u in users.values() if not u["rc"])
    vehicles    = {}
    for u in users.values():
        vehicles[u["vehicle"]] = vehicles.get(u["vehicle"], 0) + 1

    print(f"\n── Dataset summary ({'─' * 30})")
    print(f"  Total records  : {total}")
    print(f"  No Insurance   : {no_ins:4d}  ({no_ins/total*100:.1f} %)")
    print(f"  No PUC         : {no_puc:4d}  ({no_puc/total*100:.1f} %)")
    print(f"  No RC          : {no_rc:4d}  ({no_rc/total*100:.1f} %)")
    print(f"  Vehicle types  : {vehicles}")
