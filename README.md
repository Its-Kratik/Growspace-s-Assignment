# 🚀 AI/ML Assignments – GrowSpace M2M

A comprehensive AI/ML assignment showcase featuring **License Plate Detection** using classical computer vision techniques and **YOLO Bounding Box Post-Filtering** algorithms. Built with Streamlit for an interactive web experience.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-blue?style=for-the-badge&logo=streamlit)](https://growspace-s-assignment.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/Its-Kratik/Growspace-s-Assignment)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/kratik-jain12/)

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [🛠️ Technologies Used](#️-technologies-used)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📖 Usage Guide](#-usage-guide)
- [🏗️ Project Structure](#️-project-structure)
- [🔧 API Reference](#-api-reference)
- [🎨 Screenshots](#-screenshots)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👨‍💻 Author](#️-author)

## 🎯 Project Overview

This project demonstrates two key AI/ML concepts:

### 1. License Plate Detection (Classical CV)
- **No Deep Learning Required**: Uses traditional computer vision techniques
- **Edge Detection**: Canny edge detection for contour identification
- **Shape Analysis**: Filters candidates based on aspect ratio and area
- **Real-time Processing**: Interactive parameter tuning via Streamlit
- **Dataset**: Tested on [Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) from Kaggle

### 2. YOLO Bounding Box Post-Filtering
- **Non-Maximum Suppression**: Implements IoU-based filtering
- **Confidence Ranking**: Sorts detections by confidence scores
- **Rule-based Filtering**: Applies geometric constraints
- **Interactive Testing**: Real-time parameter adjustment

## ✨ Features

### 🎛️ Interactive Controls
- **Real-time Parameter Tuning**: Adjust detection parameters on-the-fly
- **Live Preview**: See results immediately as you change settings
- **Side-by-side Comparison**: Original vs. processed images
- **Responsive Design**: Works on desktop and mobile devices

### 🔍 Advanced Detection
- **Multi-scale Processing**: Handles various image sizes
- **Robust Filtering**: Multiple validation criteria
- **Error Handling**: Graceful handling of edge cases
- **Performance Optimized**: Efficient array-based processing

### 📊 Visualization
- **Bounding Box Visualization**: Clear detection overlays
- **Parameter Feedback**: Real-time validation of inputs
- **Results Display**: Formatted output with confidence scores
- **Error Messages**: User-friendly error handling

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| **Streamlit** | Latest | Web application framework |
| **OpenCV** | 4.x | Computer vision processing |
| **NumPy** | Latest | Numerical computations |
| **Pillow (PIL)** | Latest | Image processing |
| **Python** | 3.8+ | Core programming language |

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/Its-Kratik/Growspace-s-Assignment.git
cd Growspace-s-Assignment
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 🚀 Quick Start

1. **Launch the App**: Run `streamlit run app.py`
2. **Select Task**: Choose between License Plate Detection or YOLO Post-Filtering
3. **Upload Image** (Task 1): Select a vehicle image for license plate detection
4. **Adjust Parameters**: Use sliders to fine-tune detection settings
5. **View Results**: See real-time detection results with bounding boxes

## 📖 Usage Guide

### Task 1: License Plate Detection

#### Parameters Explained:
- **Canny Threshold 1**: Lower threshold for edge detection (10-200)
- **Canny Threshold 2**: Upper threshold for edge detection (50-300)
- **Minimum Area**: Minimum pixel area for valid detections (100-2000)
- **Aspect Ratio Range**: Width/height ratio constraints (1.0-6.0)

#### Best Practices:
- Start with default values and adjust gradually
- Higher Canny thresholds = fewer edges detected
- Larger minimum area = fewer false positives
- Aspect ratio should match typical license plate proportions (2.0-5.0)

### Task 2: YOLO Post-Filtering

#### Input Format:
```python
# Bounding Boxes (x1, y1, x2, y2)
[(50, 50, 200, 100), (55, 60, 190, 95), (300, 300, 305, 305)]

# Confidence Scores
[0.9, 0.85, 0.7]
```

#### Parameters Explained:
- **IoU Threshold**: Intersection over Union threshold (0.1-0.9)
- **Minimum Area**: Minimum area for valid boxes (0-2000)

## 🏗️ Project Structure

```
GrowSpace-Assignment/
├── 📁 dataset/                 # Sample images for testing
│   ├── Cars0.png
│   ├── Cars1.png
│   └── ...
├── 📄 app.py                   # Main Streamlit application
├── 📄 plate_detector.py        # License plate detection logic
├── 📄 filter_boxes.py          # YOLO post-filtering algorithms
├── 📄 utils.py                 # Utility functions (IoU calculation)
├── 📄 requirements.txt         # Python dependencies
└── 📄 README.md               # This file
```

## 🔧 API Reference

### `plate_detector.py`

#### `detect_license_plates_from_array(image_array, output_path, blur_kernel, canny_thresh, area_threshold, aspect_ratio_range)`
Processes an image array for license plate detection.

**Parameters:**
- `image_array`: Input image as numpy array (BGR format)
- `output_path`: Path to save output image
- `blur_kernel`: Gaussian blur kernel size (default: (5, 5))
- `canny_thresh`: Canny edge detection thresholds (default: (50, 150))
- `area_threshold`: Minimum area for valid detections (default: 500)
- `aspect_ratio_range`: Valid aspect ratio range (default: (2.0, 5.0))

**Returns:** Processed image array with bounding boxes

#### `preprocess_image(image, blur_kernel, canny_thresh)`
Preprocesses image for edge detection.

#### `find_plate_candidates(edged, area_threshold, aspect_ratio_range)`
Finds potential license plate regions.

### `filter_boxes.py`

#### `filter_detections(boxes, confidences, area_threshold, iou_threshold)`
Filters YOLO detections using NMS and rule-based criteria.

**Parameters:**
- `boxes`: List of bounding boxes [(x1, y1, x2, y2), ...]
- `confidences`: List of confidence scores
- `area_threshold`: Minimum area threshold
- `iou_threshold`: IoU threshold for NMS

**Returns:** List of indices for boxes to keep

### `utils.py`

#### `calculate_iou(boxA, boxB)`
Calculates Intersection over Union between two bounding boxes.

## 🎨 Screenshots

### License Plate Detection Interface
![License Plate Detection](https://via.placeholder.com/800x400/4CAF50/FFFFFF?text=License+Plate+Detection+Interface)

### YOLO Post-Filtering Interface
![YOLO Post-Filtering](https://via.placeholder.com/800x400/2196F3/FFFFFF?text=YOLO+Post-Filtering+Interface)

*Note: Replace placeholder images with actual screenshots from your application*

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest

# Format code
black .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Kratik Jain**
- 📧 Email: [kratikjain121@gmail.com](mailto:kratikjain121@gmail.com)
- 🔗 LinkedIn: [Kratik Jain](https://www.linkedin.com/in/kratik-jain12/)
- 🐙 GitHub: [@Its-Kratik](https://github.com/Its-Kratik)
- 🌐 Live Demo: [Streamlit App](https://growspace-s-assignment.streamlit.app/)

---

<div align="center">

**Made with ❤️ for GrowSpace M2M**

[![GitHub stars](https://img.shields.io/github/stars/Its-Kratik/Growspace-s-Assignment?style=social)](https://github.com/Its-Kratik/Growspace-s-Assignment)
[![GitHub forks](https://img.shields.io/github/forks/Its-Kratik/Growspace-s-Assignment?style=social)](https://github.com/Its-Kratik/Growspace-s-Assignment)

</div> 
