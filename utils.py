import cv2
import numpy as np
from typing import List, Tuple
import streamlit as st
from PIL import Image

def load_image(uploaded_file) -> np.ndarray:
    """
    Load image from Streamlit uploaded file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        OpenCV image in BGR format
    """
    # Convert PIL image to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Convert RGB to BGR if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image

def display_image_comparison(original: np.ndarray, processed: np.ndarray, title1: str = "Original", title2: str = "Processed"):
    """
    Display two images side by side in Streamlit.
    
    Args:
        original: First image
        processed: Second image
        title1: Title for first image
        title2: Title for second image
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(title1)
        # Convert BGR to RGB for display
        if len(original.shape) == 3:
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        else:
            original_rgb = original
        st.image(original_rgb, use_column_width=True)
    
    with col2:
        st.subheader(title2)
        # Convert BGR to RGB for display
        if len(processed.shape) == 3:
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        else:
            processed_rgb = processed
        st.image(processed_rgb, use_column_width=True)

def create_sample_boxes() -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    """
    Create sample bounding boxes and confidences for demonstration.
    
    Returns:
        Tuple of (boxes, confidences)
    """
    boxes = [
        (100, 100, 200, 150),  # Valid box
        (150, 120, 250, 170),  # Overlapping box
        (10, 10, 400, 20),     # Too wide
        (300, 300, 320, 315),  # Too small
        (400, 400, 500, 450),  # Valid box
        (410, 410, 510, 460),  # Overlapping with previous
        (600, 100, 800, 150),  # Valid box
    ]
    
    confidences = [0.9, 0.8, 0.95, 0.7, 0.85, 0.6, 0.75]
    
    return boxes, confidences

def visualize_boxes(image_shape: Tuple[int, int], boxes: List[Tuple[int, int, int, int]], 
                   confidences: List[float], valid_indices: List[int]) -> np.ndarray:
    """
    Create visualization of bounding boxes before and after filtering.
    
    Args:
        image_shape: Shape of the image (height, width)
        boxes: List of bounding boxes
        confidences: List of confidence scores
        valid_indices: Indices of boxes that passed filtering
        
    Returns:
        Visualization image
    """
    height, width = image_shape
    vis_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw all boxes in red (rejected)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis_image, f"{confidences[i]:.2f}", (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw valid boxes in green (accepted)
    for idx in valid_indices:
        x1, y1, x2, y2 = boxes[idx]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_image, f"{confidences[idx]:.2f}", (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return vis_image

def format_detection_results(boxes: List[Tuple[int, int, int, int]], 
                           confidences: List[float], 
                           valid_indices: List[int]) -> str:
    """
    Format detection results for display.
    
    Args:
        boxes: List of bounding boxes
        confidences: List of confidence scores
        valid_indices: Indices of valid boxes
        
    Returns:
        Formatted results string
    """
    result = f"**Original detections:** {len(boxes)}\n"
    result += f"**Valid detections:** {len(valid_indices)}\n\n"
    
    if valid_indices:
        result += "**Valid Detections:**\n"
        for i, idx in enumerate(valid_indices):
            x1, y1, x2, y2 = boxes[idx]
            width, height = x2 - x1, y2 - y1
            area = width * height
            result += f"{i+1}. Box {idx}: ({x1}, {y1}, {x2}, {y2}) - "
            result += f"Size: {width}×{height}, Area: {area}, Confidence: {confidences[idx]:.3f}\n"
    
    return result

def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU score between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # Check if there's an intersection
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate areas of both boxes
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    # Calculate IoU
    iou = intersection_area / union_area
    return iou
