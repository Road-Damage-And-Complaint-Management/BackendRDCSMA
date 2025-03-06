import streamlit as st
from ultralytics import YOLO  # type: ignore
import cv2
import numpy as np
from PIL import Image

# Load YOLO model with trained weights
model_weights_path = r"D:\Major Project Phase 2\Project Prototype-1\backend\models\best.pt"
model = YOLO(model_weights_path)

# Streamlit UI
st.title("Road Damage Detection with YOLOv8")
st.subheader("Capture an image using your webcam or upload a file to detect damages!")

# Option to choose between capturing from webcam or uploading a file
option = st.radio("Choose Input Method", ["Use Webcam", "Upload File"])

if option == "Use Webcam":
    # Capture image from webcam
    img_file_buffer = st.camera_input("Capture an image")

    if img_file_buffer:
        # Read the captured image
        image = Image.open(img_file_buffer)
        image = np.array(image)  # Convert to NumPy array

        # Ensure the image is in BGR format (YOLO uses BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Display the captured image
        st.image(image, caption="Captured Image", channels="BGR")

elif option == "Upload File":
    # Upload an image file
    img_file_buffer = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    if img_file_buffer:
        # Read the uploaded image
        image = Image.open(img_file_buffer)
        image = np.array(image)  # Convert to NumPy array

        # Ensure the image is in BGR format (YOLO uses BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", channels="BGR")

# Perform inference with YOLO (only if image is available)
if img_file_buffer:
    # Perform inference with YOLO
    results = model(image, conf=0.15)

    # Display results
    st.subheader("Detection Results")
    if len(results) > 0:
        for r in results:
            # Plot and display predictions
            plotted_image = r.plot()
            st.image(plotted_image, caption="Detected Image", use_column_width=True)
    else:
        st.write("No detections found.")