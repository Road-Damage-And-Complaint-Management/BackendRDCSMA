import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
from PIL import Image
from ultralytics import YOLO
from gps_extraction import extract_gps  # Import GPS extraction module
from admin_auth import verify_admin  # Import Admin Authentication Functions
from pymongo import MongoClient  # Import MongoDB Client
from flask import send_from_directory
import bcrypt

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017/road_damage_db"
client = MongoClient(MONGO_URI)
db = client["road_damage_db"]
reports_collection = db["reports"]  # Store uploaded image details
admin_collection = db["admins"]
# Configure session for admin login
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["road_damage_db"]
    print("✅ Connected to MongoDB successfully!")
except Exception as e:
    print(f"❌ MongoDB Connection Failed: {e}")

# Load YOLO Model
MODEL_PATH = r"C:\Users\shrey\Downloads\BackendRDCSMA-main\BackendRDCSMA-main\models\Epoch-20\best.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None  # Handle case where model fails to load

# Create upload folder if not exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/admin/login", methods=["POST"])
def admin_login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    admin = admin_collection.find_one({"email": email})

    if admin and bcrypt.checkpw(password.encode("utf-8"), admin["password"].encode("utf-8")):
        session["admin_logged_in"] = True
        return jsonify({"success": True, "message": "Login successful!"})
    else:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

# Admin Logout Route
# Logout Admin
@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.pop("admin_logged_in", None)
    return jsonify({"success": True, "message": "Logged out successfully"})

# Admin Dashboard Route
@app.route("/admin/dashboard", methods=["GET"])
def admin_dashboard():
    """Example protected route for admins."""
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized access"}), 403
    return jsonify({"message": "Welcome Admin! You have access to this route."})

@app.route("/upload", methods=["POST"])
def upload_image():
    """Handles image uploads, extracts GPS metadata, runs YOLO detection, and stores data in MongoDB."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    user_email = request.form.get("user_email", "anonymous")  # Capture user email if provided

    try:
        # Open image and convert to NumPy array for YOLO
        image = Image.open(file_path)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Perform YOLO detection
        if model is None:
            return jsonify({"error": "YOLO model failed to load"}), 500

        results = model(image_bgr, conf=0.15)
        detections = results[0] if len(results) > 0 else None

        # Extract GPS metadata
        gps_data = extract_gps(file_path)

        # Process detection results
        crack_points = len(detections.boxes) if detections else 0
        crack_type = "Detected" if crack_points > 0 else "None"

        depth_points = []
        if detections:
            for box in detections.boxes:
                depth_points.append(float(box.conf[0]))  # Adjust based on actual output format

        # Prepare data for MongoDB
        detection_data = {
            "filename": filename,
            "user_email": user_email,
            "gps_latitude": gps_data.get("gps_latitude"),
            "gps_longitude": gps_data.get("gps_longitude"),
            "location": gps_data.get("location"),
            "crack_points": crack_points,
            "crack_type": crack_type,
            "depth_points": depth_points,
            "status": "Pending"  # Default status when uploaded
        }

        # Store in MongoDB
        report = reports_collection.insert_one(detection_data)
        report_id = str(report.inserted_id)

        # Add report_id explicitly to response
        detection_data["_id"] = report_id

        return jsonify({
            "message": "✅ Image uploaded successfully!",
            "report_id": report_id,
            "data": detection_data
        })

    except Exception as e:
        return jsonify({"error": f"❌ Processing failed: {str(e)}"}), 500

@app.route("/reports", methods=["GET"])
def fetch_reports():
    """Fetch all reports from the database."""
    try:
        reports_cursor = reports_collection.find({}, {"_id": 1, "filename": 1, "gps_latitude": 1, "gps_longitude": 1, "location": 1, "crack_points": 1, "crack_type": 1, "depth_points": 1, "status": 1, "user_email": 1})

        reports = []
        for report in reports_cursor:
            report["_id"] = str(report["_id"])  # Convert ObjectId to string
            reports.append(report)

        return jsonify({"reports": reports})
    except Exception as e:
        return jsonify({"error": f"❌ Failed to fetch reports: {str(e)}"}), 500

@app.route("/report/<report_id>", methods=["GET"])
def get_report(report_id):
    """Fetch a specific report by ID to track its status."""
    try:
        report = reports_collection.find_one({"_id": report_id})
        if not report:
            return jsonify({"error": "Report not found"}), 404

        report["_id"] = str(report["_id"])
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": f"❌ Failed to fetch report: {str(e)}"}), 500

@app.route("/update_status/<report_id>", methods=["POST"])
def update_status(report_id):
    """Update the status of a report (e.g., Pending → In Progress → Resolved)."""
    data = request.json
    new_status = data.get("status")

    if not new_status:
        return jsonify({"error": "Status is required"}), 400

    try:
        result = reports_collection.update_one({"_id": report_id}, {"$set": {"status": new_status}})
        if result.modified_count == 0:
            return jsonify({"error": "Report not found or status unchanged"}), 404

        return jsonify({"message": "✅ Status updated successfully!", "report_id": report_id, "new_status": new_status})
    except Exception as e:
        return jsonify({"error": f"❌ Failed to update status: {str(e)}"}), 500

@app.route("/user_reports", methods=["GET"])
def user_reports():
    """Fetch reports for a specific user."""
    user_email = request.args.get("user_email")
    if not user_email:
        return jsonify({"error": "User email is required"}), 400

    reports = list(reports_collection.find({"user_email": user_email}))
    for report in reports:
        report["_id"] = str(report["_id"])

    return jsonify({"reports": reports})


if __name__ == "__main__":
    app.run(debug=True)

