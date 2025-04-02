from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
DETECTED_FOLDER = "detected"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

model_weights_path = "best.pt"  # Adjust the path accordingly
model = YOLO(model_weights_path)

@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Load and process image
    image = Image.open(filepath)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Perform inference with YOLO
    results = model(image)

    if results:
        plotted_image = results[0].plot()
        detected_path = os.path.join(DETECTED_FOLDER, file.filename)
        cv2.imwrite(detected_path, plotted_image)  # Save detected image

        return jsonify({
            "original_image": f"/uploads/{file.filename}",
            "detected_image": f"/detected/{file.filename}",
            "crack_points": len(results[0].boxes)  # Example of extracting detected cracks
        })

    return jsonify({"error": "No detections found"}), 200

@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/detected/<filename>")
def get_detected_file(filename):
    return send_from_directory(DETECTED_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
