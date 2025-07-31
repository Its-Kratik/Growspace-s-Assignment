from typing import List, Tuple
import numpy as np

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

def filter_detections(
    boxes: List[Tuple[int, int, int, int]],  # (x1, y1, x2, y2)
    confidences: List[float]
) -> List[int]:  # Indices of valid boxes
    """
    Filter YOLO detections based on aspect ratio, area, and IoU-based suppression.
    
    Args:
        boxes: List of bounding boxes in format (x1, y1, x2, y2)
        confidences: List of confidence scores corresponding to each box
        
    Returns:
        List of indices of valid boxes after filtering
    """
    if len(boxes) != len(confidences):
        raise ValueError("Number of boxes must match number of confidences")
    
    if len(boxes) == 0:
        return []
    
    valid_indices = []
    
    # Step 1: Filter by geometric constraints
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Calculate width, height, and area
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Apply filtering rules
        # Rule 1a: Discard if width > 3 * height
        if width > 3 * height:
            continue
            
        # Rule 1b: Discard if area < 400 pixels
        if area < 400:
            continue
            
        valid_indices.append(i)
    
    # Step 2: Apply IoU-based Non-Maximum Suppression
    if len(valid_indices) <= 1:
        return valid_indices
    
    # Sort indices by confidence scores in descending order
    valid_indices.sort(key=lambda i: confidences[i], reverse=True)
    
    final_indices = []
    
    # Apply NMS with IoU threshold of 0.4
    for i in range(len(valid_indices)):
        current_idx = valid_indices[i]
        keep_box = True
        
        # Check against all previously selected boxes
        for selected_idx in final_indices:
            iou = compute_iou(boxes[current_idx], boxes[selected_idx])
            
            # If IoU > 0.4, suppress the current box (keep the higher confidence one)
            if iou > 0.4:
                keep_box = False
                break
        
        if keep_box:
            final_indices.append(current_idx)
    
    return final_indices

def test_filter_detections():
    """
    Test function to demonstrate the filter_detections function.
    """
    # Test case 1: Basic filtering
    boxes = [
        (100, 100, 200, 150),  # Valid box
        (150, 120, 250, 170),  # Overlapping box (should be suppressed)
        (10, 10, 400, 20),     # Too wide (width > 3 * height)
        (300, 300, 320, 315),  # Too small area (20 * 15 = 300 < 400)
        (400, 400, 500, 450),  # Valid box
    ]
    
    confidences = [0.9, 0.8, 0.95, 0.7, 0.85]
    
    print("Test Case 1:")
    print(f"Input boxes: {len(boxes)}")
    print(f"Input confidences: {confidences}")
    
    result = filter_detections(boxes, confidences)
    print(f"Valid indices: {result}")
    print(f"Number of valid detections: {len(result)}")
    
    # Print details of valid detections
    for idx in result:
        x1, y1, x2, y2 = boxes[idx]
        width, height = x2 - x1, y2 - y1
        area = width * height
        print(f"  Box {idx}: ({x1}, {y1}, {x2}, {y2}) - "
              f"Width: {width}, Height: {height}, Area: {area}, "
              f"Confidence: {confidences[idx]:.2f}")
    
    print()
    
    # Test case 2: IoU suppression
    boxes2 = [
        (100, 100, 200, 150),  # High confidence
        (110, 110, 210, 160),  # High overlap with first box
        (300, 300, 400, 350),  # Separate box
    ]
    
    confidences2 = [0.9, 0.8, 0.7]
    
    print("Test Case 2 (IoU Suppression):")
    print(f"Input boxes: {len(boxes2)}")
    
    result2 = filter_detections(boxes2, confidences2)
    print(f"Valid indices: {result2}")
    
    # Calculate IoU between first two boxes
    iou = compute_iou(boxes2[0], boxes2[1])
    print(f"IoU between box 0 and box 1: {iou:.3f}")
    
    print()

if __name__ == "__main__":
    test_filter_detections()
