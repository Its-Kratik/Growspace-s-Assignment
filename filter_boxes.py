from typing import List, Tuple
from utils import calculate_iou
def calculate_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Each box is in format (x1, y1, x2, y2)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea

    if unionArea == 0:
        return 0.0

    return interArea / unionArea

def filter_detections(
    boxes: List[Tuple[int, int, int, int]],
    confidences: List[float],
    area_threshold: int = 400,
    iou_threshold: float = 0.4
) -> List[int]:
    """
    Filters boxes based on shape rules and IoU-based suppression.
    Returns indices of boxes to keep.
    """
    assert len(boxes) == len(confidences), "Box and confidence lists must match"

    # Rule-based filtering
    valid = []
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        width = x2 - x1
        height = y2 - y1
        area = width * height
        if width > 3 * height or area < area_threshold:
            continue
        valid.append(idx)

    # Sort remaining boxes by confidence
    valid = sorted(valid, key=lambda i: confidences[i], reverse=True)

    keep = []
    removed = set()

    for i in valid:
        if i in removed:
            continue
        keep.append(i)
        for j in valid:
            if i == j or j in removed:
                continue
            iou = calculate_iou(boxes[i], boxes[j])
            if iou > iou_threshold:
                removed.add(j)

    return keep
