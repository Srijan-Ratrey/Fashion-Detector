"""
Utility functions for Fashion AI Video Analysis
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import pandas as pd


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fashion_ai.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_vibes_list(vibes_path: str = "data/vibeslist.json") -> List[str]:
    """Load the predefined vibes list"""
    with open(vibes_path, 'r') as f:
        vibes = json.load(f)
    return vibes


def load_product_catalog(catalog_path: str = "data/images.csv") -> pd.DataFrame:
    """Load product catalog from CSV"""
    return pd.read_csv(catalog_path)


def create_output_dir(output_path: str) -> None:
    """Create output directory if it doesn't exist"""
    os.makedirs(output_path, exist_ok=True)


def save_json_output(data: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file"""
    create_output_dir(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def extract_video_id(video_path: str) -> str:
    """Extract video ID from file path"""
    return Path(video_path).stem


def crop_bounding_box(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop image using bounding box coordinates (x, y, w, h)"""
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize image to target size"""
    return cv2.resize(image, target_size)


def preprocess_for_clip(image: np.ndarray) -> Image.Image:
    """Preprocess image for CLIP model"""
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    return Image.fromarray(image_rgb)


def calculate_confidence_score(similarity: float) -> str:
    """Calculate match type based on similarity score"""
    if similarity > 0.9:
        return "exact"
    elif similarity >= 0.75:
        return "similar"
    else:
        return "no_match"


def filter_detections_by_confidence(detections: List[Dict], min_confidence: float = 0.5) -> List[Dict]:
    """Filter detections based on minimum confidence threshold"""
    return [det for det in detections if det.get('confidence', 0) >= min_confidence]


def parse_color_from_text(text: str) -> Optional[str]:
    """Extract color from text description"""
    colors = [
        'black', 'white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple',
        'pink', 'brown', 'gray', 'grey', 'gold', 'silver', 'beige', 'navy',
        'maroon', 'olive', 'lime', 'aqua', 'teal', 'fuchsia', 'tan', 'khaki'
    ]
    
    text_lower = text.lower()
    for color in colors:
        if color in text_lower:
            return color
    return None


def validate_output_format(output_data: Dict[str, Any]) -> bool:
    """Validate output JSON format"""
    required_fields = ['video_id', 'vibes', 'products']
    
    # Check required fields
    for field in required_fields:
        if field not in output_data:
            return False
    
    # Validate vibes format
    if not isinstance(output_data['vibes'], list):
        return False
    
    # Validate products format
    if not isinstance(output_data['products'], list):
        return False
    
    # Validate each product
    for product in output_data['products']:
        required_product_fields = ['type', 'color', 'matched_product_id', 'match_type', 'confidence']
        for field in required_product_fields:
            if field not in product:
                return False
    
    return True


def get_video_files(video_dir: str) -> List[str]:
    """Get all video files from directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f'*{ext}'))
    
    return [str(f) for f in video_files]


class Config:
    """Configuration class for the fashion AI pipeline"""
    
    # Paths
    DATA_DIR = "data"
    MODELS_DIR = "models"
    OUTPUT_DIR = "outputs"
    
    # Model settings
    YOLO_MODEL = "yolov8n.pt"
    CLIP_MODEL = "ViT-B/32"
    WHISPER_MODEL = "base"
    
    # Detection settings
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_SIMILARITY_THRESHOLD = 0.75
    
    # Processing settings
    MAX_FRAMES_PER_VIDEO = 30
    FRAME_EXTRACTION_FPS = 2
    
    # Fashion categories
    FASHION_CLASSES = [
        'person', 'handbag', 'tie', 'suitcase', 'backpack',
        # Custom fashion classes would be added here
    ]
    
    @classmethod
    def load_from_file(cls, config_path: str):
        """Load configuration from JSON file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    setattr(cls, key.upper(), value)


def benchmark_processing_time(func):
    """Decorator to benchmark processing time"""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper 