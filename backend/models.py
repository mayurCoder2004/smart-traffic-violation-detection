import uuid
from datetime import datetime, timezone, timedelta
from .extensions import db

IST = timezone(timedelta(hours=5, minutes=30))


def _ist_iso(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).isoformat()


def _uuid():
    return str(uuid.uuid4())


class User(db.Model):
    __tablename__ = "users"

    id            = db.Column(db.String(36),  primary_key=True, default=_uuid)
    name          = db.Column(db.String(100), nullable=False)
    phone         = db.Column(db.String(20),  nullable=False)
    license_plate = db.Column(db.String(20),  unique=True, nullable=False, index=True)
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)

    # Vehicle & document fields — seeded from users.json via seed_db.py
    vehicle   = db.Column(db.String(20),  nullable=True,  default="Unknown")
    rc        = db.Column(db.Boolean,     nullable=False, default=True)
    insurance = db.Column(db.Boolean,     nullable=False, default=True)
    puc       = db.Column(db.Boolean,     nullable=False, default=True)
    violations = db.relationship("Violation", backref="user", lazy=True)
    scanner_challans = db.relationship("ScannerChallan", backref="user", lazy=True)

    def to_dict(self):
        return {
            "id":            self.id,
            "name":          self.name,
            "phone":         self.phone,
            "license_plate": self.license_plate,
            "vehicle":       self.vehicle,
            "documents": {
                "rc":        self.rc,
                "insurance": self.insurance,
                "puc":       self.puc,

            },
            "created_at":    _ist_iso(self.created_at),
        }

    def to_scan_dict(self):
        """Compact response shape for the /scan/<plate> endpoint."""
        return {
            "found":   True,
            "plate":   self.license_plate,
            "owner":   self.name,
            "vehicle": self.vehicle or "Unknown",
            "phone":   self.phone,
            "documents": {
                "rc":        self.rc,
                "insurance": self.insurance,
                "puc":       self.puc,

            },
            "source": "postgresql",
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
            "created_at":     _ist_iso(self.created_at),
        }
        if include_user and self.user:
            d["user"] = {"name": self.user.name, "phone": self.user.phone}
        else:
            d["user"] = None
        return d


class ScannerChallan(db.Model):
    __tablename__ = "scanner_challans"

    id           = db.Column(db.String(20), primary_key=True)
    user_id      = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=True)
    plate_number = db.Column(db.String(20), nullable=False, index=True)
    owner_name   = db.Column(db.String(100), nullable=False)
    vehicle      = db.Column(db.String(20), nullable=False, default="Unknown")
    fine_amount  = db.Column(db.Integer, nullable=False, default=0)
    status       = db.Column(db.String(20), nullable=False, default="UNPAID")
    created_at   = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    items = db.relationship(
        "ScannerChallanItem",
        backref="challan",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="ScannerChallanItem.id",
    )

    def to_dict(self):
        return {
            "id":         self.id,
            "plate":      self.plate_number,
            "owner":      self.owner_name,
            "vehicle":    self.vehicle,
            "violations": [item.to_dict() for item in self.items],
            "fine":       self.fine_amount,
            "status":     self.status,
            "timestamp":  _ist_iso(self.created_at),
        }


class ScannerChallanItem(db.Model):
    __tablename__ = "scanner_challan_items"

    id         = db.Column(db.Integer, primary_key=True)
    challan_id = db.Column(db.String(20), db.ForeignKey("scanner_challans.id"), nullable=False, index=True)
    type       = db.Column(db.String(80), nullable=False)
    fine       = db.Column(db.Integer, nullable=False)
    source     = db.Column(db.String(20), nullable=False)

    def to_dict(self):
        return {
            "type":   self.type,
            "fine":   self.fine,
            "source": self.source,
        }


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
            "created_at":          _ist_iso(self.created_at),
        }
