from flask import Blueprint, jsonify, request
from ..models import User
from ..extensions import db

users_bp = Blueprint("users", __name__, url_prefix="/users")


@users_bp.route("", methods=["POST"])
def create_user():
    data = request.get_json()
    if not data or not all(k in data for k in ("name", "phone", "license_plate")):
        return jsonify({"error": "name, phone, and license_plate are required"}), 400

    plate = data["license_plate"].strip().upper()

    if User.query.filter_by(license_plate=plate).first():
        return jsonify({"error": "License plate already registered"}), 409

    user = User(name=data["name"].strip(), phone=data["phone"].strip(), license_plate=plate)
    db.session.add(user)
    db.session.commit()
    return jsonify(user.to_dict()), 201


@users_bp.route("/login", methods=["POST"])
def login():
    """Simple plate-based login — returns user info if found."""
    data = request.get_json()
    if not data or "license_plate" not in data:
        return jsonify({"error": "license_plate is required"}), 400

    user = User.query.filter_by(license_plate=data["license_plate"].strip().upper()).first()
    if not user:
        return jsonify({"error": "No account found for this plate"}), 404

    return jsonify(user.to_dict())


@users_bp.route("/<plate_number>", methods=["GET"])
def get_user_by_plate(plate_number):
    user = User.query.filter_by(license_plate=plate_number.upper()).first()
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user.to_dict())


@users_bp.route("", methods=["GET"])
def list_users():
    users = User.query.order_by(User.created_at.desc()).all()
    return jsonify([u.to_dict() for u in users])
