import uuid
from datetime import datetime
from .extensions import db


def _uuid():
    return str(uuid.uuid4())


class User(db.Model):
    __tablename__ = "users"

    id            = db.Column(db.String(36), primary_key=True, default=_uuid)
    name          = db.Column(db.String(100), nullable=False)
    phone         = db.Column(db.String(20),  nullable=False)
    license_plate = db.Column(db.String(20),  unique=True, nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    violations = db.relationship("Violation", backref="user", lazy=True)

    def to_dict(self):
        return {
            "id":            self.id,
            "name":          self.name,
            "phone":         self.phone,
            "license_plate": self.license_plate,
            "created_at":    self.created_at.isoformat(),
        }


class Violation(db.Model):
    __tablename__ = "violations"

    id             = db.Column(db.String(36), primary_key=True, default=_uuid)
    user_id        = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=True)
    plate_number   = db.Column(db.String(20), nullable=False)
    violation_type = db.Column(db.String(50), nullable=False)
    fine_amount    = db.Column(db.Integer,    nullable=False)
    status         = db.Column(db.String(20), default="PENDING")
    created_at     = db.Column(db.DateTime,   default=datetime.utcnow)

    payments = db.relationship("Payment", backref="violation", lazy=True)

    def to_dict(self, include_user=False):
        d = {
            "id":             self.id,
            "plate_number":   self.plate_number,
            "violation_type": self.violation_type,
            "fine_amount":    self.fine_amount,
            "status":         self.status,
            "created_at":     self.created_at.isoformat(),
        }
        if include_user and self.user:
            d["user"] = {"name": self.user.name, "phone": self.user.phone}
        else:
            d["user"] = None
        return d


class Payment(db.Model):
    __tablename__ = "payments"

    id                  = db.Column(db.String(36), primary_key=True, default=_uuid)
    violation_id        = db.Column(db.String(36), db.ForeignKey("violations.id"), nullable=False)
    razorpay_order_id   = db.Column(db.String(100), nullable=True)
    razorpay_payment_id = db.Column(db.String(100), nullable=True)
    amount              = db.Column(db.Integer, nullable=False)
    status              = db.Column(db.String(20), default="PENDING")
    created_at          = db.Column(db.DateTime,   default=datetime.utcnow)

    def to_dict(self):
        return {
            "id":                  self.id,
            "violation_id":        self.violation_id,
            "razorpay_order_id":   self.razorpay_order_id,
            "razorpay_payment_id": self.razorpay_payment_id,
            "amount":              self.amount,
            "status":              self.status,
            "created_at":          self.created_at.isoformat(),
        }
