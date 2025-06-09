#!/usr/bin/env python3
"""
Fashion item detection using YOLOv8
"""

import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
import colorsys

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Run: pip install ultralytics")
    YOLO = None

# Try to import utils, fall back to basic implementations if not available
try:
    from .utils import benchmark_processing_time, Config, filter_detections_by_confidence
except ImportError:
    # Fallback implementations
    def benchmark_processing_time(func):
        """Simple benchmark decorator fallback"""
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    
    class Config:
        """Simple config class fallback"""
        pass
    
    def filter_detections_by_confidence(detections, threshold):
        """Simple confidence filtering fallback"""
        return [d for d in detections if d.get('confidence', 0) >= threshold]

logger = logging.getLogger(__name__)


class FashionDetector:
    """YOLOv8-based fashion item detector"""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize FashionDetector
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()
        
        # Fashion-related class mappings (COCO dataset classes)
        self.fashion_classes = {
            # Main fashion items
            0: 'person',  # Will be analyzed for clothing
            27: 'tie',
            28: 'suitcase', 
            31: 'backpack',
            33: 'handbag',
        }
        
        # Custom fashion categories mapping
        self.fashion_categories = {
            'person': ['top', 'bottom', 'dress', 'outerwear'],
            'handbag': ['bag'],
            'tie': ['accessory'],
            'suitcase': ['bag'],
            'backpack': ['bag'],
        }
        
        # Clothing type detection regions (relative to person bounding box)
        self.clothing_regions = {
            'top': (0.0, 0.0, 1.0, 0.4),  # Upper 40% of person
            'bottom': (0.0, 0.4, 1.0, 0.6),  # Middle 20% of person
            'dress': (0.0, 0.0, 1.0, 0.6),  # Upper 60% of person
            'outerwear': (0.0, 0.0, 1.0, 0.5)  # Upper 50% of person
        }
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            if YOLO is None:
                raise ImportError("ultralytics package not available")
                
            self.model = YOLO(self.model_path)
            logger.info(f"Loaded YOLO model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    @benchmark_processing_time
    def detect_fashion_items(self, frame: np.ndarray, frame_number: int = 0) -> List[Dict[str, Any]]:
        """
        Detect fashion items in a single frame
        
        Args:
            frame: Input frame as numpy array
            frame_number: Frame number for tracking
            
        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Extract box information
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Convert to x, y, w, h format
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        
                        # Get class name
                        class_name = self.model.names.get(class_id, f"class_{class_id}")
                        
                        # Check if it's a fashion-related class
                        if self._is_fashion_class(class_id, class_name):
                            if class_name == 'person':
                                # For person detections, analyze clothing regions
                                clothing_items = self._analyze_person_clothing(frame, (x, y, w, h))
                                detections.extend(clothing_items)
                            else:
                                # For other fashion items (accessories, bags, etc.)
                                detection = {
                                    'class_name': class_name,
                                    'class_id': class_id,
                                    'bbox': (x, y, w, h),
                                    'confidence': confidence,
                                    'frame_number': frame_number,
                                    'fashion_type': self._get_fashion_type(class_name),
                                    'color': self._estimate_dominant_color(frame[y:y+h, x:x+w])
                                }
                                detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} fashion items in frame {frame_number}")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed for frame {frame_number}: {e}")
            return []
    
    def _analyze_person_clothing(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """
        Analyze clothing items on detected person
        
        Args:
            frame: Input frame
            person_bbox: Person bounding box (x, y, w, h)
            
        Returns:
            List of detected clothing items
        """
        x, y, w, h = person_bbox
        clothing_items = []
        
        # Analyze each clothing region
        for clothing_type, (rx, ry, rw, rh) in self.clothing_regions.items():
            # Calculate region coordinates
            region_x = x + int(rx * w)
            region_y = y + int(ry * h)
            region_w = int(rw * w)
            region_h = int(rh * h)
            
            # Ensure coordinates are within frame bounds
            frame_h, frame_w = frame.shape[:2]
            region_x = max(0, min(region_x, frame_w - 1))
            region_y = max(0, min(region_y, frame_h - 1))
            region_w = min(region_w, frame_w - region_x)
            region_h = min(region_h, frame_h - region_y)
            
            if region_w > 0 and region_h > 0:
                # Extract region
                region = frame[region_y:region_y+region_h, region_x:region_x+region_w]
                
                # Create detection for this clothing item
                clothing_item = {
                    'class_name': clothing_type,
                    'bbox': (region_x, region_y, region_w, region_h),
                    'confidence': 0.8,  # Default confidence for clothing regions
                    'fashion_type': clothing_type,
                    'color': self._estimate_dominant_color(region)
                }
                clothing_items.append(clothing_item)
        
        return clothing_items
    
    def _estimate_dominant_color(self, image_region: np.ndarray) -> str:
        """
        Estimate dominant color in image region
        
        Args:
            image_region: Image region to analyze
            
        Returns:
            Dominant color name
        """
        # Convert to RGB and calculate mean color
        if len(image_region.shape) == 3:
            mean_color = np.mean(image_region, axis=(0, 1))
            b, g, r = mean_color
            
            # Simple color classification
            colors = {
                'black': (0, 0, 0),
                'white': (255, 255, 255),
                'red': (255, 0, 0),
                'green': (0, 255, 0),
                'blue': (0, 0, 255),
                'yellow': (255, 255, 0),
                'purple': (128, 0, 128),
                'orange': (255, 165, 0),
                'pink': (255, 192, 203),
                'brown': (165, 42, 42)
            }
            
            # Find closest color
            min_distance = float('inf')
            closest_color = 'unknown'
            
            for color_name, (cr, cg, cb) in colors.items():
                distance = np.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_color = color_name
            
            return closest_color
        
        return 'unknown'
    
    def _is_fashion_class(self, class_id: int, class_name: str) -> bool:
        """Check if detected class is fashion-related"""
        # Check by class ID
        if class_id in self.fashion_classes:
            return True
            
        # Check by class name - expanded list
        fashion_keywords = [
            'person', 'handbag', 'tie', 'suitcase', 'backpack',
            'bag', 'purse', 'shoe', 'boot', 'hat', 'cap', 'sunglasses',
            'watch', 'belt', 'scarf', 'jewelry', 'earring', 'necklace',
            'clothing', 'shirt', 'dress', 'pants', 'skirt', 'jacket',
            'coat', 'sweater', 'top', 'bottom', 'accessory'
        ]
        
        for keyword in fashion_keywords:
            if keyword in class_name.lower():
                return True
                
        return False
    
    def _get_fashion_type(self, class_name: str) -> str:
        """Map detected class to fashion type"""
        class_to_type = {
            'person': 'clothing',  # Will be further analyzed
            'handbag': 'bag',
            'tie': 'accessory',
            'suitcase': 'bag',
            'backpack': 'bag',
            'umbrella': 'accessory',
            'shoe': 'footwear',
            'boot': 'footwear',
            'hat': 'headwear',
            'cap': 'headwear'
        }
        
        return class_to_type.get(class_name.lower(), 'unknown')
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Detect fashion items in multiple frames
        
        Args:
            frames: List of frames as numpy arrays
            
        Returns:
            List of detection lists (one per frame)
        """
        all_detections = []
        
        for i, frame in enumerate(frames):
            detections = self.detect_fashion_items(frame, frame_number=i)
            all_detections.append(detections)
        
        return all_detections
    
    def crop_detected_items(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Crop detected fashion items from frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            List of (cropped_image, detection_info) tuples
        """
        cropped_items = []
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            
            # Ensure coordinates are within frame bounds
            frame_h, frame_w = frame.shape[:2]
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = min(w, frame_w - x)
            h = min(h, frame_h - y)
            
            if w > 0 and h > 0:
                cropped = frame[y:y+h, x:x+w].copy()
                cropped_items.append((cropped, detection))
        
        return cropped_items
    
    def filter_overlapping_detections(self, detections: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Filter overlapping detections using Non-Maximum Suppression
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Filtered detections
        """
        if not detections:
            return []
        
        # Convert to format suitable for NMS
        boxes = []
        scores = []
        
        for det in detections:
            x, y, w, h = det['bbox']
            boxes.append([x, y, x + w, y + h])
            scores.append(det['confidence'])
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            self.confidence_threshold, 
            iou_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        
        return []
    
    def get_detection_summary(self, all_detections: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Get summary statistics of detections across all frames
        
        Args:
            all_detections: List of detection lists from all frames
            
        Returns:
            Summary statistics
        """
        total_detections = sum(len(dets) for dets in all_detections)
        
        if total_detections == 0:
            return {'total_detections': 0, 'classes': {}, 'confidence_stats': {}}
        
        # Count detections by class
        class_counts = {}
        confidences = []
        
        for frame_detections in all_detections:
            for det in frame_detections:
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                confidences.append(det['confidence'])
        
        # Calculate confidence statistics
        confidence_stats = {
            'mean': np.mean(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'std': np.std(confidences)
        }
        
        return {
            'total_detections': total_detections,
            'frames_processed': len(all_detections),
            'classes': class_counts,
            'confidence_stats': confidence_stats
        } 