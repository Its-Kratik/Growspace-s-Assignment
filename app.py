import sys
if 'cv2' in sys.modules:
    del sys.modules['cv2']
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


# Import our custom modules
from plate_detector import LicensePlateDetector
from filter_boxes import filter_detections
from utils import (
    load_image, 
    display_image_comparison, 
    create_sample_boxes, 
    visualize_boxes, 
    format_detection_results,
    compute_iou
)

# Set page configuration
st.set_page_config(
    page_title="AI/ML Assignment - Computer Vision App",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🚗 AI/ML Assignment - Computer Vision App")
    st.markdown("""
    This application demonstrates two computer vision tasks:
    1. **License Plate Detection** using classical image processing
    2. **YOLO Bounding Box Post-Filtering** with IoU-based suppression
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    task_choice = st.sidebar.selectbox(
        "Choose a task:",
        ["📋 Overview", "🎯 Task 1: License Plate Detection", "📦 Task 2: YOLO Box Filtering"]
    )
    
    if task_choice == "📋 Overview":
        show_overview()
    elif task_choice == "🎯 Task 1: License Plate Detection":
        show_license_plate_detection()
    elif task_choice == "📦 Task 2: YOLO Box Filtering":
        show_yolo_filtering()

def show_overview():
    """Display overview of the assignment."""
    
    st.header("📋 Assignment Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Task 1: License Plate Detection")
        st.markdown("""
        **Objective:** Detect license plate regions using classical image processing
        
        **Requirements:**
        - Convert image to grayscale
        - Find rectangular regions with high-contrast edges
        - Filter by aspect ratio (2:1 to 5:1)
        - No deep learning models allowed
        
        **Techniques Used:**
        - Gaussian blur for noise reduction
        - Bilateral filtering
        - Canny edge detection
        - Morphological operations (closing, dilation)
        - Contour analysis
        """)
    
    with col2:
        st.subheader("📦 Task 2: YOLO Box Post-Filtering")
        st.markdown("""
        **Objective:** Filter YOLO detection results using geometric and IoU constraints
        
        **Filtering Rules:**
        1. Discard boxes where width > 3 × height
        2. Discard boxes with area < 400 pixels
        3. Apply IoU-based Non-Maximum Suppression (threshold: 0.4)
        
        **Techniques Used:**
        - Aspect ratio filtering
        - Area-based filtering
        - Intersection over Union (IoU) calculation
        - Non-Maximum Suppression (NMS)
        """)
    
    # Technical specifications
    st.subheader("🛠️ Technical Specifications")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Libraries Used", "OpenCV, NumPy", "No Deep Learning")
    
    with col2:
        st.metric("Image Processing", "Classical CV", "Morphological Ops")
    
    with col3:
        st.metric("Filtering Method", "Geometric + IoU", "NMS Algorithm")

def show_license_plate_detection():
    """Display license plate detection interface."""
    
    st.header("🎯 Task 1: License Plate Detection")
    st.markdown("""
    Upload an image containing vehicles to detect license plate regions using classical image processing techniques.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image containing one or more vehicles"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        image = load_image(uploaded_file)
        
        st.subheader("📷 Original Image")
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(original_rgb, caption="Input Image", use_container_width=True)
        
        # Processing parameters in sidebar
        st.sidebar.markdown("### 🔧 Detection Parameters")
        min_aspect_ratio = st.sidebar.slider("Minimum Aspect Ratio", 1.5, 3.0, 2.0, 0.1)
        max_aspect_ratio = st.sidebar.slider("Maximum Aspect Ratio", 3.0, 7.0, 5.0, 0.1)
        min_area = st.sidebar.slider("Minimum Area (pixels)", 200, 1000, 400, 50)
        
        # Create detector with custom parameters
        detector = LicensePlateDetector()
        detector.min_aspect_ratio = min_aspect_ratio
        detector.max_aspect_ratio = max_aspect_ratio
        detector.min_area = min_area
        
        # Process button
        if st.button("🔍 Detect License Plates", type="primary"):
            with st.spinner("Processing image..."):
                # Detect license plates
                detections = detector.detect_license_plates(image)
                
                # Draw detections
                result_image = detector.draw_detections(image, detections)
                
                # Display results
                st.subheader(f"🎯 Detection Results ({len(detections)} plates found)")
                
                if len(detections) > 0:
                    # Display comparison
                    display_image_comparison(image, result_image, "Original", "Detected Plates")
                    
                    # Show detection details
                    st.subheader("📊 Detection Details")
                    for i, (x, y, w, h) in enumerate(detections):
                        area = w * h
                        aspect_ratio = w / h
                        st.write(f"**Plate {i+1}:** Position: ({x}, {y}), Size: {w}×{h}, "
                               f"Area: {area} px, Aspect Ratio: {aspect_ratio:.2f}")
                    
                    # Provide download link for result
                    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    result_pil = Image.fromarray(result_rgb)
                    buf = io.BytesIO()
                    result_pil.save(buf, format='JPEG')
                    
                    st.download_button(
                        label="📥 Download Result",
                        data=buf.getvalue(),
                        file_name="license_plate_detection_result.jpg",
                        mime="image/jpeg"
                    )
                else:
                    st.warning("No license plates detected. Try adjusting the parameters.")
    else:
        # Show example
        st.info("👆 Upload an image to start detection")
        
        # Display example preprocessing steps
        st.subheader("🔬 Processing Pipeline")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**1. Grayscale**")
            st.markdown("Convert to grayscale and apply Gaussian blur")
        
        with col2:
            st.markdown("**2. Edge Detection**")
            st.markdown("Apply bilateral filter and Canny edge detection")
        
        with col3:
            st.markdown("**3. Morphology**")
            st.markdown("Use closing and dilation to connect characters")
        
        with col4:
            st.markdown("**4. Filtering**")
            st.markdown("Filter by aspect ratio and area constraints")

def show_yolo_filtering():
    """Display YOLO bounding box filtering interface."""
    
    st.header("📦 Task 2: YOLO Bounding Box Post-Filtering")
    st.markdown("""
    Demonstrate filtering of YOLO detection results using geometric constraints and IoU-based suppression.
    """)
    
    # Option to use sample data or upload custom data
    data_source = st.radio(
        "Data Source:",
        ["🎯 Use Sample Data", "📁 Upload Custom Data"]
    )
    
    if data_source == "🎯 Use Sample Data":
        # Use predefined sample boxes
        boxes, confidences = create_sample_boxes()
        st.success(f"Loaded {len(boxes)} sample bounding boxes")
        
    else:
        st.info("Custom data upload feature - paste your YOLO results below:")
        
        # Text area for custom boxes
        boxes_text = st.text_area(
            "Bounding Boxes (x1,y1,x2,y2 per line):",
            value="100,100,200,150\n150,120,250,170\n10,10,400,20",
            height=100
        )
        
        confidences_text = st.text_area(
            "Confidences (one per line):",
            value="0.9\n0.8\n0.95",
            height=100
        )
        
        try:
            # Parse custom data
            boxes = []
            for line in boxes_text.strip().split('\n'):
                if line.strip():
                    coords = [int(x.strip()) for x in line.split(',')]
                    boxes.append(tuple(coords))
            
            confidences = []
            for line in confidences_text.strip().split('\n'):
                if line.strip():
                    confidences.append(float(line.strip()))
            
            if len(boxes) != len(confidences):
                st.error("Number of boxes must match number of confidences!")
                return
                
        except Exception as e:
            st.error(f"Error parsing data: {e}")
            return
    
    # Display original detections
    st.subheader("📊 Original Detections")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create visualization
        vis_image = visualize_boxes((600, 900), boxes, confidences, list(range(len(boxes))))
        vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        st.image(vis_rgb, caption="All Detections (Red: Original)", use_container_width=True)
    
    with col2:
        st.markdown("**Detection Summary:**")
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            area = width * height
            st.write(f"**Box {i}:** {width}×{height} (Area: {area}) - Conf: {conf:.2f}")
    
    # Apply filtering
    if st.button("🔍 Apply Filtering", type="primary"):
        with st.spinner("Applying filters..."):
            # Apply the filtering function
            valid_indices = filter_detections(boxes, confidences)
            
            st.subheader("✅ Filtering Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create filtered visualization
                filtered_vis = visualize_boxes((600, 900), boxes, confidences, valid_indices)
                filtered_rgb = cv2.cvtColor(filtered_vis, cv2.COLOR_BGR2RGB)
                st.image(filtered_rgb, caption="After Filtering (Green: Kept, Red: Rejected)", use_container_width=True)
            
            with col2:
                # Display results
                results_text = format_detection_results(boxes, confidences, valid_indices)
                st.markdown(results_text)
            
            # Show detailed analysis
            st.subheader("🔬 Filtering Analysis")
            
            # Rule 1: Aspect ratio filtering
            aspect_ratio_failures = []
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                width, height = x2 - x1, y2 - y1
                if width > 3 * height:
                    aspect_ratio_failures.append(i)
            
            if aspect_ratio_failures:
                st.write(f"**Aspect Ratio Filter:** Rejected boxes {aspect_ratio_failures} (width > 3 × height)")
            
            # Rule 2: Area filtering
            area_failures = []
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                width, height = x2 - x1, y2 - y1
                area = width * height
                if area < 400:
                    area_failures.append(i)
            
            if area_failures:
                st.write(f"**Area Filter:** Rejected boxes {area_failures} (area < 400 pixels)")
            
            # Rule 3: IoU-based suppression
            st.write("**IoU-based Suppression:**")
            suppressed = []
            for i in range(len(boxes)):
                if i not in valid_indices and i not in aspect_ratio_failures and i not in area_failures:
                    suppressed.append(i)
            
            if suppressed:
                st.write(f"Suppressed boxes {suppressed} due to high IoU overlap")
                
                # Show IoU matrix
                if st.checkbox("Show IoU Matrix"):
                    st.write("**IoU Matrix:**")
                    iou_matrix = np.zeros((len(boxes), len(boxes)))
                    for i in range(len(boxes)):
                        for j in range(len(boxes)):
                            if i != j:
                                iou_matrix[i][j] = compute_iou(boxes[i], boxes[j])
                    
                    st.dataframe(
                        iou_matrix,
                        use_container_width=True
                    )
            else:
                st.write("No boxes were suppressed by IoU filtering")

if __name__ == "__main__":
    main()
