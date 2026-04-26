from datetime import datetime
from flask import Blueprint, jsonify, request
from ..models import User, Violation
from ..extensions import db

violations_bp = Blueprint("violations", __name__, url_prefix="/violations")

FINE_AMOUNTS = {
    "helmet":    500,
    "triple":   1000,
    "overspeed": 1000,
}

# In-memory per-plate cooldown to avoid duplicate entries from the CV pipeline.
# Key: (plate, vtype) → datetime of last recorded violation.
_cooldown: dict = {}
COOLDOWN_SECS = 10


@violations_bp.route("", methods=["POST"])
def create_violation():
    data = request.get_json()
    if not data or not all(k in data for k in ("plate_number", "violation_type")):
        return jsonify({"error": "plate_number and violation_type are required"}), 400

    plate = data["plate_number"].strip().upper()
    vtype = data["violation_type"].strip().lower()

    if vtype not in FINE_AMOUNTS:
        return jsonify({"error": f"violation_type must be one of: {', '.join(FINE_AMOUNTS)}"}), 400

    # Cooldown gate — prevents duplicate entries within COOLDOWN_SECS seconds
    key      = (plate, vtype)
    last     = _cooldown.get(key)
    now      = datetime.utcnow()
    if last and (now - last).total_seconds() < COOLDOWN_SECS:
        return jsonify({"skipped": True, "reason": "cooldown active"}), 200

    _cooldown[key] = now

    user = User.query.filter_by(license_plate=plate).first()

    violation = Violation(
        user_id        = user.id if user else None,
        plate_number   = plate,
        violation_type = vtype,
        fine_amount    = FINE_AMOUNTS[vtype],
        status         = "PENDING",
    )
    db.session.add(violation)
    db.session.commit()

    result = violation.to_dict(include_user=True)
    return jsonify(result), 201


@violations_bp.route("", methods=["GET"])
def get_all_violations():
    violations = Violation.query.order_by(Violation.created_at.desc()).all()
    return jsonify([v.to_dict(include_user=True) for v in violations])


@violations_bp.route("/user/<user_id>", methods=["GET"])
def get_user_violations(user_id):
    violations = (
        Violation.query
        .filter_by(user_id=user_id)
        .order_by(Violation.created_at.desc())
        .all()
    )
    return jsonify([v.to_dict() for v in violations])


@violations_bp.route("/<violation_id>", methods=["GET"])
def get_violation(violation_id):
    v = db.session.get(Violation, violation_id)
    if not v:
        return jsonify({"error": "Violation not found"}), 404
    return jsonify(v.to_dict(include_user=True))
