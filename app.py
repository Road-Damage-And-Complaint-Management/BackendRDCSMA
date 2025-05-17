import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_session import Session
from PIL import Image
from ultralytics import YOLO
from gps_extraction import extract_gps  # GPS metadata extraction
from pymongo import MongoClient  # MongoDB integration

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017/road_damage_db"
client = MongoClient(MONGO_URI)
db = client["road_damage_db"]
reports_collection = db["reports"]

# Configure session
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Ensure MongoDB is connected
try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB successfully!")
except Exception as e:
    print(f"❌ MongoDB Connection Failed: {e}")

# Load YOLO Model
MODEL_PATH = r"D:\Major Project Phase 2\Project Prototype-1\backend\models\Epoch-20\best.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None  # Handle case where model fails to load

# Create upload folder if not exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_image():
    """Handles image uploads, extracts GPS metadata, runs YOLO detection, and stores data in MongoDB."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    user_email = request.form.get("user_email", "anonymous")

    try:
        image = Image.open(file_path)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        if model is None:
            return jsonify({"error": "YOLO model failed to load"}), 500

        results = model(image_bgr, conf=0.15)
        detections = results[0] if len(results) > 0 else None

        # Generate detected image with bounding boxes
        detected_image_filename = f"detected_{filename}"
        detected_image_path = os.path.join(UPLOAD_FOLDER, detected_image_filename)

        if detections:
            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

            cv2.imwrite(detected_image_path, image_bgr)  # Save detected image
        else:
            detected_image_path = file_path  # If no detections, use original

        # Extract GPS data
        gps_data = extract_gps(file_path)

        crack_points = len(detections.boxes) if detections else 0
        crack_type = "Detected" if crack_points > 0 else "None"

        # Save to MongoDB
        detection_data = {
            "original_image": filename,
            "detected_image": detected_image_filename,
            "user_email": user_email,
            "gps_latitude": gps_data.get("gps_latitude", "Unknown"),
            "gps_longitude": gps_data.get("gps_longitude", "Unknown"),
            "location": gps_data.get("location", "Unknown Location"),
            "crack_points": crack_points,
            "crack_type": crack_type,
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

@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    """Serves uploaded images to the frontend."""
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
