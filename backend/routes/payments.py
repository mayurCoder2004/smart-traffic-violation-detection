import hmac
import hashlib
from flask import Blueprint, jsonify, request, current_app
from ..models import Violation, Payment
from ..extensions import db

try:
    import razorpay as _razorpay_lib
    _RAZORPAY_AVAILABLE = True
except ImportError:
    _RAZORPAY_AVAILABLE = False

payments_bp = Blueprint("payments", __name__, url_prefix="/payments")


def _client():
    key_id     = current_app.config.get("RAZORPAY_KEY_ID", "")
    key_secret = current_app.config.get("RAZORPAY_KEY_SECRET", "")
    if not key_id or not key_secret or not _RAZORPAY_AVAILABLE:
        return None
    return _razorpay_lib.Client(auth=(key_id, key_secret))


@payments_bp.route("/create-order", methods=["POST"])
def create_order():
    data         = request.get_json() or {}
    violation_id = data.get("violation_id")

    violation = db.session.get(Violation, violation_id)
    if not violation:
        return jsonify({"error": "Violation not found"}), 404
    if violation.status == "PAID":
        return jsonify({"error": "Already paid"}), 400

    client = _client()
    if not client:
        return jsonify({"error": "Payment gateway not configured — add RAZORPAY keys to .env"}), 503

    order = client.order.create({
        "amount":   violation.fine_amount * 100,   # amount in paise
        "currency": "INR",
        "receipt":  f"challan_{violation_id[:8]}",
        "notes": {
            "violation_id":    violation_id,
            "plate_number":    violation.plate_number,
            "violation_type":  violation.violation_type,
        },
    })

    payment = Payment(
        violation_id      = violation_id,
        razorpay_order_id = order["id"],
        amount            = violation.fine_amount,
        status            = "PENDING",
    )
    db.session.add(payment)
    db.session.commit()

    return jsonify({
        "order_id":  order["id"],
        "amount":    violation.fine_amount,
        "currency":  "INR",
        "key_id":    current_app.config.get("RAZORPAY_KEY_ID"),
        "violation": {
            "id":    violation.id,
            "type":  violation.violation_type,
            "plate": violation.plate_number,
        },
    })


@payments_bp.route("/verify", methods=["POST"])
def verify_payment():
    data = request.get_json() or {}

    order_id     = data.get("razorpay_order_id", "")
    payment_id   = data.get("razorpay_payment_id", "")
    signature    = data.get("razorpay_signature", "")
    violation_id = data.get("violation_id", "")

    key_secret = current_app.config.get("RAZORPAY_KEY_SECRET", "")

    # Verify HMAC-SHA256 signature
    message  = f"{order_id}|{payment_id}"
    expected = hmac.new(
        key_secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if expected != signature:
        return jsonify({"error": "Invalid payment signature"}), 400

    # Update payment record
    payment = Payment.query.filter_by(razorpay_order_id=order_id).first()
    if payment:
        payment.razorpay_payment_id = payment_id
        payment.status              = "PAID"

    # Mark violation as paid
    violation = db.session.get(Violation, violation_id)
    if violation:
        violation.status = "PAID"

    db.session.commit()
    return jsonify({"success": True, "message": "Payment recorded successfully"})
