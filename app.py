
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from plate_detector import detect_license_plates
from filter_boxes import filter_detections

st.set_page_config(page_title="AI/ML Assignments – GrowSpace", layout="centered")
st.title("🚀 AI/ML Assignments – GrowSpace M2M")

task = st.sidebar.selectbox("📂 Select Assignment", [
    "License Plate Detection (Classical CV)",
    "YOLO Box Post-Filtering"
])

# Task 1: License Plate Detection
if task.startswith("License Plate"):
    st.header("📸 Task 1: License Plate Detection (No Deep Learning)")

    uploaded_file = st.file_uploader("Upload an image of a vehicle", type=["jpg", "jpeg", "png"])

    canny1 = st.slider("Canny Threshold 1", 10, 200, 50)
    canny2 = st.slider("Canny Threshold 2", 50, 300, 150)
    min_area = st.slider("Minimum Area (px²)", 100, 2000, 500)
    aspect_low, aspect_high = st.slider("Aspect Ratio Range", 1.0, 6.0, (2.0, 5.0))

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_input.write(uploaded_file.read())
        temp_input.flush()

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        detect_license_plates(
            input_path=temp_input.name,
            output_path=temp_output.name,
            canny_thresh=(canny1, canny2),
            area_threshold=min_area,
            aspect_ratio_range=(aspect_low, aspect_high)
        )

        col1, col2 = st.columns(2)
        col1.image(img, caption="Uploaded Image", use_column_width=True)
        col2.image(temp_output.name, caption="Detected Plates", use_column_width=True)

# Task 2: YOLO Post-Filtering
else:
    st.header("🧠 Task 2: YOLO Bounding Box Post-Filtering")

    st.markdown("### ✏️ Input your YOLO detections")

    default_boxes = "[(50, 50, 200, 100), (55, 60, 190, 95), (300, 300, 305, 305)]"
    default_confs = "[0.9, 0.85, 0.7]"

    boxes_text = st.text_area("🔲 Bounding Boxes (x1, y1, x2, y2)", default_boxes, height=120)
    confs_text = st.text_area("📊 Confidence Scores", default_confs, height=80)

    iou_thresh = st.slider("IoU Threshold", 0.1, 0.9, 0.4)
    min_area = st.slider("Minimum Area (px²)", 0, 2000, 400)

    if st.button("🧹 Run Post-Filtering"):
        try:
            boxes = eval(boxes_text)
            confs = eval(confs_text)

            indices = filter_detections(
                boxes=boxes,
                confidences=confs,
                area_threshold=min_area,
                iou_threshold=iou_thresh
            )

            st.success(f"✅ Valid box indices: {indices}")
            st.write("🎯 Final Boxes:")
            for idx in indices:
                st.write(f"{idx}: Box={boxes[idx]}, Confidence={confs[idx]}")

        except Exception as e:
            st.error(f"❌ Error in parsing input: {e}")
