import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_session import Session
from PIL import Image
from ultralytics import YOLO
from gps_extraction import extract_gps
from pymongo import MongoClient
import bcrypt
from bson import ObjectId
# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Configure session
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["road_damage_db"]
reports_collection = db["reports"]
admin_collection = db["admins"]

# Ensure MongoDB is connected
try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB successfully!")
except Exception as e:
    print(f"❌ MongoDB Connection Failed: {e}")

# ========== ADMIN LOGIN SETUP ==========
def create_admin():
    email = "admin4@example.com"
    password = "admin1233"
    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    if admin_collection.find_one({"email": email}):
        print("✅ Admin already exists.")
    else:
        admin_collection.insert_one({"email": email, "password": hashed_pw})
        print("✅ Admin created successfully.")

@app.route("/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    admin = admin_collection.find_one({"email": email})
    if admin and bcrypt.checkpw(password.encode("utf-8"), admin["password"]):
        return jsonify({"success": True, "message": "Login successful"})
    else:
        return jsonify({"success": False, "message": "Invalid email or password"})

# ========== YOLO SETUP ==========
MODEL_PATH = r"C:\Users\shrey\Documents\majorProject\Road_Damage_detection\BackendRDCSMA-main\models\Epoch-20\best.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========== ROUTES ==========
@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    user_name = request.form.get("user_email", "anonymous")  # ✅ Grab name from form

    try:
        image = Image.open(file_path)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        if model is None:
            return jsonify({"error": "YOLO model failed to load"}), 500

        results = model(image_bgr, conf=0.15)
        detections = results[0] if len(results) > 0 else None

        detected_image_filename = f"detected_{filename}"
        detected_image_path = os.path.join(UPLOAD_FOLDER, detected_image_filename)

        depth_points = []

        if detections and detections.boxes:
            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf_value = float(box.conf[0])
                depth_points.append(conf_value)

            cv2.imwrite(detected_image_path, image_bgr)
        else:
            detected_image_path = file_path

        gps_data = extract_gps(file_path)

        crack_points = len(detections.boxes) if detections and detections.boxes else 0
        crack_type = "Detected" if crack_points > 0 else "None"

        detection_data = {
            "original_image": filename,
            "detected_image": detected_image_filename,
            "user_name": user_name,  # ✅ Save to DB
            "gps_latitude": gps_data.get("gps_latitude", "Unknown"),
            "gps_longitude": gps_data.get("gps_longitude", "Unknown"),
            "location": gps_data.get("location", "Unknown Location"),
            "crack_points": crack_points,
            "crack_type": crack_type,
            "depth_points": depth_points,
            "status": "Pending"
        }

        report = reports_collection.insert_one(detection_data)
        report_id = str(report.inserted_id)
        detection_data["_id"] = report_id

        return jsonify({
            "message": "✅ Image uploaded successfully!",
            "report_id": report_id,
            "data": detection_data
        })

    except Exception as e:
        return jsonify({"error": f"❌ Processing failed: {str(e)}"}), 500
def get_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/reports", methods=["GET"])
def fetch_reports():
    try:
        reports_cursor = reports_collection.find({}, {"_id": 1, "filename": 1, "gps_latitude": 1, "gps_longitude": 1, "location": 1, "crack_points": 1, "crack_type": 1, "status": 1, "user_email": 1})

        reports = []
        for report in reports_cursor:
            report["_id"] = str(report["_id"])
            reports.append(report)

        return jsonify({"reports": reports})
    except Exception as e:
        return jsonify({"error": f"❌ Failed to fetch reports: {str(e)}"}), 500
@app.route("/report/<report_id>", methods=["GET"])
def get_report(report_id):
    try:
        report = reports_collection.find_one({"_id": ObjectId(report_id)})
        if not report:
            return jsonify({"error": "Report not found"}), 404

        report["_id"] = str(report["_id"])

        # Ensure depth_points is returned as a list (or empty list if missing)
        if "depth_points" not in report or not isinstance(report["depth_points"], list):
            report["depth_points"] = []

        return jsonify(report)
    except Exception as e:
        return jsonify({"error": f"❌ Failed to fetch report: {str(e)}"}), 500


@app.route("/update_status/<report_id>", methods=["POST"])
def update_status(report_id):
    data = request.json
    new_status = data.get("status")

    if not new_status:
        return jsonify({"error": "Status is required"}), 400

    try:
        result = reports_collection.update_one(
            {"_id": ObjectId(report_id)},
            {"$set": {"status": new_status}}
        )
        if result.matched_count == 0:
            return jsonify({"error": "Report not found"}), 404
        if result.modified_count == 0:
            return jsonify({"message": "Status unchanged"}), 200

        return jsonify({
            "message": "✅ Status updated successfully!",
            "report_id": report_id,
            "new_status": new_status
        })
    except Exception as e:
        return jsonify({"error": f"❌ Failed to update status: {str(e)}"}), 500


@app.route("/user_reports", methods=["GET"])
def user_reports():
    user_email = request.args.get("user_email")
    if not user_email:
        return jsonify({"error": "User email is required"}), 400

    try:
        reports = list(reports_collection.find({"user_email": user_email}))
        for report in reports:
            report["_id"] = str(report["_id"])
            if "depth_points" not in report or not isinstance(report["depth_points"], list):
                report["depth_points"] = []

        return jsonify({"reports": reports})
    except Exception as e:
        return jsonify({"error": f"❌ Failed to fetch user reports: {str(e)}"}), 500
if __name__ == "__main__":
    create_admin()  # Ensure admin is created on first run
    app.run(debug=True, port=5000)
