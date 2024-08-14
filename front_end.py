import cv2
import time
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Function to navigate to the detect page
def go_to_detect():
    st.session_state.page = "detect"

# Function to navigate to the upload page
def go_to_upload():
    st.session_state.page = "upload"

def load_model():
    model = YOLO("D:/Pg-DAI/CV/car_damage_detection/yolov8/models/best_model.pt")
    return model

# Function to process the image and make predictions
def detect_damage_and_draw_boxes(image, model):
    # Perform inference
    results = model(image)

    # Convert PIL image to a NumPy array (OpenCV format)
    image_cv = np.array(image)

    # Access class names directly from the model
    labels = model.names  # Access class names

    # Draw bounding boxes on the image
    for detection in results[0].boxes:
        # Extract bounding box coordinates and other attributes
        box = detection.xyxy[0].tolist()  # Ensure correct unpacking of xyxy
        conf = detection.conf[0].tolist() if detection.conf is not None else 0
        cls = int(detection.cls[0].tolist()) if detection.cls is not None else 0

        if len(box) == 4:  # If the result contains bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            label = labels[cls] if labels else str(cls)
            color = (0, 255, 0)  # Green color for bounding boxes
            # Draw bounding box
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
            # Draw label and confidence
            cv2.putText(image_cv, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert NumPy array back to PIL Image for displaying in Streamlit
    image_annotated = Image.fromarray(image_cv)

    # Prepare results DataFrame for display
    detections_df = []
    for detection in results[0].boxes:
        box = detection.xyxy[0].tolist()
        conf = detection.conf[0].tolist() if detection.conf is not None else 0
        cls = int(detection.cls[0].tolist()) if detection.cls is not None else 0

        if len(box) == 4:  # Ensure the box contains 4 coordinates
            x1, y1, x2, y2 = map(int, box)
            detections_df.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'confidence': conf,
                'class': labels[cls] if labels else str(cls)
            })

    return image_annotated, detections_df

# Initialize session state if not already done
if "page" not in st.session_state:
    st.session_state.page = "upload"

# Load the model
model = load_model()

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# Check the current page
if st.session_state.page == "upload":

    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown("<h1>Image Upload and Display</h1>", unsafe_allow_html=True)

        with col2:
            if st.button("Detect Damage"):
                if st.session_state.uploaded_file is not None:
                    go_to_detect()
                else:
                    warning_placeholder = st.empty()
                    warning_placeholder.write("Please upload an image file first!!")
                    time.sleep(3)
                    warning_placeholder.empty()

    st.session_state.uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if st.session_state.uploaded_file is not None:
        image = Image.open(st.session_state.uploaded_file).convert("RGB")
        st.session_state.image = image

        st.write("Image has been successfully uploaded and displayed.")
        st.image(image, caption="Uploaded Image.", use_column_width=True)

elif st.session_state.page == "detect":
    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown("<h1>Damage Detection Page</h1>", unsafe_allow_html=True)

        with col2:
            if st.button("Go Back to the Upload"):
                go_to_upload()

    if st.session_state.uploaded_file is not None:
        image_annotated, detections_df = detect_damage_and_draw_boxes(st.session_state.image, model)

        st.image(image_annotated, caption="Detected Image with Bounding Boxes", use_column_width=True)
        st.write("Predictions:")
        st.write(detections_df)
        #st.image(st.session_state.image, caption="Uploaded Image.", use_column_width=True)
    else:
        st.write("No image uploaded yet!")
