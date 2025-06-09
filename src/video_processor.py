"""
Video processing module for extracting frames from fashion videos
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import os

from .utils import benchmark_processing_time, Config

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video frame extraction and preprocessing"""
    
    def __init__(self, fps: int = 2, max_frames: int = 30):
        """
        Initialize VideoProcessor
        
        Args:
            fps: Frames per second to extract
            max_frames: Maximum number of frames to extract per video
        """
        self.fps = fps
        self.max_frames = max_frames
        
    @benchmark_processing_time
    def extract_frames(self, video_path: str, output_dir: Optional[str] = None) -> List[np.ndarray]:
        """
        Extract key frames from video
        
        Args:
            video_path: Path to input video file
            output_dir: Optional directory to save extracted frames
            
        Returns:
            List of extracted frames as numpy arrays
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
            
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Original FPS: {original_fps}, Duration: {duration:.2f}s, Total frames: {total_frames}")
        
        # Calculate frame sampling interval
        frame_interval = max(1, int(original_fps / self.fps)) if original_fps > 0 else 1
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Extract frame at specified intervals
                if frame_count % frame_interval == 0 and extracted_count < self.max_frames:
                    frames.append(frame.copy())
                    
                    # Save frame if output directory specified
                    if output_dir:
                        self._save_frame(frame, output_dir, video_path, extracted_count)
                    
                    extracted_count += 1
                    logger.debug(f"Extracted frame {extracted_count} at {frame_count}/{total_frames}")
                    
                frame_count += 1
                
        finally:
            cap.release()
            
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def _save_frame(self, frame: np.ndarray, output_dir: str, video_path: str, frame_index: int):
        """Save extracted frame to disk"""
        os.makedirs(output_dir, exist_ok=True)
        video_name = Path(video_path).stem
        frame_filename = f"{video_name}_frame_{frame_index:04d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
    
    def extract_keyframes_uniform(self, video_path: str, num_frames: int = 10) -> List[np.ndarray]:
        """
        Extract uniformly distributed keyframes from video
        
        Args:
            video_path: Path to input video
            num_frames: Number of frames to extract
            
        Returns:
            List of extracted keyframes
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= num_frames:
            # If video has fewer frames than requested, extract all
            return self.extract_frames(video_path)
        
        # Calculate frame indices to extract
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        try:
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame.copy())
                    
        finally:
            cap.release()
            
        logger.info(f"Extracted {len(frames)} uniform keyframes from {video_path}")
        return frames
    
    def extract_scene_changes(self, video_path: str, threshold: float = 0.3) -> List[np.ndarray]:
        """
        Extract frames at scene changes using histogram comparison
        
        Args:
            video_path: Path to input video
            threshold: Threshold for scene change detection
            
        Returns:
            List of frames at scene changes
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
            
        frames = []
        prev_hist = None
        frame_count = 0
        
        try:
            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate histogram for current frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                curr_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                
                if prev_hist is not None:
                    # Compare histograms using correlation
                    correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
                    
                    # If correlation is below threshold, it's a scene change
                    if correlation < (1 - threshold):
                        frames.append(frame.copy())
                        logger.debug(f"Scene change detected at frame {frame_count}, correlation: {correlation:.3f}")
                
                prev_hist = curr_hist
                frame_count += 1
                
        finally:
            cap.release()
            
        # Ensure we have at least one frame
        if not frames:
            # Fallback to uniform extraction
            frames = self.extract_keyframes_uniform(video_path, min(5, self.max_frames))
            
        logger.info(f"Extracted {len(frames)} frames at scene changes from {video_path}")
        return frames
    
    def preprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Preprocess frame for object detection
        
        Args:
            frame: Input frame
            target_size: Target size for resizing
            
        Returns:
            Preprocessed frame
        """
        # Resize while maintaining aspect ratio
        h, w = frame.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create padded frame
        processed = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the resized frame
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        processed[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return processed
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
            
        try:
            info = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': 0,
                'codec': None
            }
            
            # Calculate duration
            if info['fps'] > 0:
                info['duration'] = info['frame_count'] / info['fps']
                
            # Get codec information
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            info['codec'] = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
            
        finally:
            cap.release()
            
        return info
    
    def extract_audio_transcript(self, video_path: str) -> Optional[str]:
        """
        Extract audio and convert to text using Whisper
        Note: This is a placeholder - actual implementation would use Whisper
        
        Args:
            video_path: Path to video file
            
        Returns:
            Transcript text or None if audio extraction fails
        """
        try:
            # This would integrate with Whisper for actual audio transcription
            # For now, return None as placeholder
            logger.info(f"Audio transcript extraction not implemented for {video_path}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract audio transcript: {e}")
            return None


def process_video_batch(video_paths: List[str], output_dir: str, processor: VideoProcessor) -> dict:
    """
    Process multiple videos in batch
    
    Args:
        video_paths: List of video file paths
        output_dir: Output directory for frames
        processor: VideoProcessor instance
        
    Returns:
        Dictionary mapping video paths to extracted frames
    """
    results = {}
    
    for video_path in video_paths:
        try:
            video_output_dir = os.path.join(output_dir, Path(video_path).stem)
            frames = processor.extract_frames(video_path, video_output_dir)
            results[video_path] = frames
            
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            results[video_path] = []
            
    return results 