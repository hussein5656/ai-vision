"""Simplified detector using YOLO's native tracking."""

import cv2
import numpy as np
from typing import Optional, List, Dict, Any
import os

# COCO class names (80 classes)
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Vehicle classes we care about for parking logic (COCO ids)
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}


def get_class_name(class_id: Optional[int]) -> str:
    """Return human-readable class name for a detection."""
    if class_id is None:
        return "objet"
    try:
        if 0 <= int(class_id) < len(COCO_CLASS_NAMES):
            return COCO_CLASS_NAMES[int(class_id)]
    except Exception:
        pass
    return f"classe {class_id}"


def is_vehicle_class(class_id: Optional[int]) -> bool:
    """Check if class ID corresponds to a vehicle we track for parking."""
    if class_id is None:
        return False
    try:
        return int(class_id) in VEHICLE_CLASS_IDS
    except Exception:
        return False


class Detector:
    """Detector using YOLO client for detection and YOLO's native tracking."""
    
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.4, max_side: Optional[int] = None):
        self.conf = conf
        self.model_path = os.path.abspath(os.path.expanduser(model_path or "yolov8n.pt"))
        self.max_side = max_side
        self.yolo_client = None
        self.bg_sub = None
        self._init_yolo_client()
    
    def _init_yolo_client(self):
        """Start YOLO client in subprocess."""
        try:
            from vision.yolo_client import YOLOClient
            self.yolo_client = YOLOClient(self.model_path)
            if self.yolo_client.is_ready():
                print("[Detector] YOLO detection ready (with native tracking)")
            else:
                self.yolo_client = None
        except Exception as e:
            print(f"[Detector] YOLO unavailable: {e}, using motion detection fallback")
            self.yolo_client = None
        
        # Initialize fallback for when YOLO isn't available
        if self.yolo_client is None:
            try:
                self.bg_sub = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=16, detectShadows=False
                )
                print("[Detector] Motion detection fallback initialized")
            except Exception:
                self.bg_sub = None
    
    def detect(self, frame, count_classes=None, conf=None, max_side: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Detect objects in frame. Returns list with track_id, class, bbox, center, confidence.
        
        Uses YOLO's native tracking if available, else motion detection.
        YOLO already provides track_id via persist=True, so we don't need additional tracking.
        """
        resize_meta = None
        target_side = max_side or self.max_side
        if target_side:
            resize_meta, frame = self._resize_for_detection(frame, target_side)

        # Prefer YOLO (which has native tracking)
        if self.yolo_client and self.yolo_client.is_ready():
            try:
                detections = self.yolo_client.detect(frame)
                # Filter by class if specified
                if count_classes is not None and detections:
                    detections = [d for d in detections if d.get('class', 0) in count_classes]
                if resize_meta:
                    detections = self._rescale_detections(detections, resize_meta)
                return detections
            except Exception as e:
                print(f"[Detector] YOLO error: {e}")
        
        # Fallback: motion detection
        return self._detect_motion(frame, count_classes)
    
    def _detect_motion(self, frame, count_classes=None) -> List[Dict[str, Any]]:
        """Detect motion using background subtraction."""
        detections = []
        if self.bg_sub is None:
            return detections
        
        try:
            fg = self.bg_sub.apply(frame)
            fg = cv2.medianBlur(fg, 5)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < 400:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                detections.append({
                    "track_id": i,  # Simple index-based ID for motion
                    "class": -1,
                    "bbox": (x, y, x + w, y + h),
                    "center": (cx, cy),
                    "confidence": 0.5,
                })
        except Exception:
            pass
        
        return detections
    
    def shutdown(self):
        """Cleanup resources."""
        if self.yolo_client:
            self.yolo_client.shutdown()

    def set_max_side(self, value: Optional[int]):
        self.max_side = value

    def _resize_for_detection(self, frame, target_side: int):
        if frame is None:
            return None, frame
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        if max_dim <= target_side:
            return None, frame
        scale = target_side / float(max_dim)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return {'scale_x': w / float(new_w), 'scale_y': h / float(new_h)}, resized

    def _rescale_detections(self, detections: List[Dict[str, Any]], meta: Dict[str, float]):
        scale_x = meta.get('scale_x', 1.0)
        scale_y = meta.get('scale_y', 1.0)
        if not detections or (scale_x == 1.0 and scale_y == 1.0):
            return detections
        for det in detections:
            bbox = det.get('bbox')
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                det['bbox'] = (
                    int(round(x1 * scale_x)),
                    int(round(y1 * scale_y)),
                    int(round(x2 * scale_x)),
                    int(round(y2 * scale_y))
                )
            center = det.get('center')
            if center and len(center) == 2:
                cx, cy = center
                det['center'] = (
                    int(round(cx * scale_x)),
                    int(round(cy * scale_y))
                )
        return detections
