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

def check_plate_like_shape(contour, tolerance=0.1):
    """
    Check if contour has a rectangular shape similar to license plates.
    """
    # Approximate the contour to a polygon
    epsilon = tolerance * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Check if it's approximately a rectangle (4 sides)
    if len(approx) == 4:
        return True
    return False

def check_edge_density(edged, x, y, w, h, min_density=0.1):
    """
    Check if the region has sufficient edge density (characteristic of text).
    """
    roi = edged[y:y+h, x:x+w]
    if roi.size == 0:
        return False
    
    # Calculate edge density
    edge_pixels = np.count_nonzero(roi)
    total_pixels = roi.size
    density = edge_pixels / total_pixels
    
    return density >= min_density

def find_plate_candidates(edged, area_threshold=500, aspect_ratio_range=(2.0, 5.0), edge_density_threshold=0.1):
    """
    Find contours and filter based on shape, aspect ratio, and texture.
    """
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    
    print(f"🔍 Processing {len(contours)} total contours...")

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Basic validation
        if not is_valid_plate(w, h, area_threshold, aspect_ratio_range):
            continue
            
        # Advanced shape validation
        if not check_plate_like_shape(cnt):
            continue
            
        # Edge density check for text-like regions
        if not check_edge_density(edged, x, y, w, h, edge_density_threshold):
            continue
            
        candidates.append((x, y, w, h))

    print(f"✅ Found {len(candidates)} valid plate-like regions.")
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

def detect_license_plates_from_array(image_array, output_path="output.jpg",
                          blur_kernel=(5, 5), canny_thresh=(50, 150),
                          area_threshold=500, aspect_ratio_range=(2.0, 5.0), edge_density_threshold=0.1):
    """
    Directly detect from an image array (used in Streamlit)
    """
    image = image_array.copy()
    edged = preprocess_image(image, blur_kernel, canny_thresh)
    candidates = find_plate_candidates(edged, area_threshold, aspect_ratio_range, edge_density_threshold)
    result = draw_bounding_boxes(image, candidates)
    cv2.imwrite(output_path, result)
    return result
