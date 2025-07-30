import cv2
import numpy as np

def preprocess_image(image, blur_kernel=(5, 5), canny_thresh=(50, 150)):
    """
    Convert image to grayscale, blur it, and apply Canny edge detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    edged = cv2.Canny(blurred, canny_thresh[0], canny_thresh[1])
    return edged

def is_valid_plate(w, h, area_threshold=500, aspect_ratio_range=(2.0, 5.0)):
    """
    Validate based on area and aspect ratio.
    """
    area = w * h
    if area < area_threshold:
        return False
    aspect_ratio = w / float(h)
    return aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]

def find_plate_candidates(edged, area_threshold=500, aspect_ratio_range=(2.0, 5.0)):
    """
    Find contours and filter based on shape and aspect ratio.
    """
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if is_valid_plate(w, h, area_threshold, aspect_ratio_range):
            candidates.append((x, y, w, h))

    return candidates

def draw_bounding_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """
    Draw rectangles around valid candidates.
    """
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image

def detect_license_plates(input_path, output_path="output.jpg",
                          blur_kernel=(5, 5), canny_thresh=(50, 150),
                          area_threshold=500, aspect_ratio_range=(2.0, 5.0)):
    """
    End-to-end pipeline to detect license plate-like regions.
    """
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {input_path}")

    edged = preprocess_image(image, blur_kernel, canny_thresh)
    candidates = find_plate_candidates(edged, area_threshold, aspect_ratio_range)
    result = draw_bounding_boxes(image, candidates)
    cv2.imwrite(output_path, result)
