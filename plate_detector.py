import cv2
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class LicensePlateDetector:
    """
    Classical license plate detection using image processing techniques
    without deep learning models.
    """
    
    def __init__(self):
        self.min_aspect_ratio = 2.0
        self.max_aspect_ratio = 5.0
        self.min_area = 400
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale and apply preprocessing.
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def detect_edges(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Apply edge detection to find high-contrast regions.
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            Binary edge image
        """
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray_image, 11, 17, 17)
        
        # Apply Canny edge detection
        edges = cv2.Canny(filtered, 30, 200)
        
        return edges
    
    def apply_morphological_operations(self, edges: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to connect license plate characters.
        
        Args:
            edges: Binary edge image
            
        Returns:
            Processed binary image
        """
        # Create rectangular kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        
        # Apply closing operation to connect text characters
        morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Apply dilation to make the regions more solid
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        morph = cv2.dilate(morph, kernel2, iterations=1)
        
        return morph
    
    def find_license_plate_candidates(self, processed_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Find rectangular regions that could be license plates.
        
        Args:
            processed_image: Binary processed image
            
        Returns:
            List of bounding box coordinates (x, y, w, h)
        """
        # Find contours
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate area and aspect ratio
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter based on aspect ratio and area
            if (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio and 
                area >= self.min_area):
                candidates.append((x, y, w, h))
        
        return candidates
    
    def detect_license_plates(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Main detection pipeline for license plates.
        
        Args:
            image: Input BGR image
            
        Returns:
            List of detected license plate bounding boxes (x, y, w, h)
        """
        # Preprocess image
        gray = self.preprocess_image(image)
        
        # Detect edges
        edges = self.detect_edges(gray)
        
        # Apply morphological operations
        processed = self.apply_morphological_operations(edges)
        
        # Find candidates
        candidates = self.find_license_plate_candidates(processed)
        
        return candidates
    
    def draw_detections(self, image: np.ndarray, detections: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Draw bounding boxes around detected license plates.
        
        Args:
            image: Input image
            detections: List of bounding box coordinates
            
        Returns:
            Image with drawn bounding boxes
        """
        result = image.copy()
        
        for (x, y, w, h) in detections:
            # Draw rectangle
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(result, 'License Plate', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result

def main():
    """
    Main function to run license plate detection.
    """
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python plate_detector.py <input_image_path>")
        return
    
    input_path = sys.argv[1]
    
    try:
        # Load input image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not load image from {input_path}")
            return
        
        # Initialize detector
        detector = LicensePlateDetector()
        
        # Detect license plates
        detections = detector.detect_license_plates(image)
        
        print(f"Found {len(detections)} potential license plates")
        
        # Draw detections
        result = detector.draw_detections(image, detections)
        
        # Save output
        output_path = "output.jpg"
        cv2.imwrite(output_path, result)
        print(f"Result saved to {output_path}")
        
        # Display result (if running in interactive environment)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title(f'Detected License Plates ({len(detections)})')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        except:
            pass
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
