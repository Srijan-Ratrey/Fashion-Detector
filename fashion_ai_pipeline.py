#!/usr/bin/env python3
"""
Complete Fashion AI Pipeline for Flickd Hackathon
Optimized for production with efficient catalog management
"""

import os
import sys
import json
import cv2
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our components
import fashion_detector
import product_matcher
import vibe_classifier

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FashionAIPipeline:
    """Complete Fashion AI Pipeline for video processing"""
    
    def __init__(self, catalog_size: int = 1500):
        """
        Initialize the pipeline
        
        Args:
            catalog_size: Number of products to use from catalog (for performance)
        """
        self.catalog_size = catalog_size
        self.detector = None
        self.matcher = None
        self.classifier = None
        self._setup_components()
    
    def _setup_components(self):
        """Initialize all pipeline components"""
        logger.info("🚀 Initializing Fashion AI Pipeline...")
        
        try:
            # Create optimized catalog
            self._create_optimized_catalog()
            
            # Initialize Fashion Detector with lower confidence for better detection
            logger.info("📷 Loading Fashion Detector...")
            self.detector = fashion_detector.FashionDetector(confidence_threshold=0.25)
            logger.info("✅ Fashion Detector loaded")
            
            # Initialize Product Matcher with optimized catalog
            logger.info("🔍 Loading Product Matcher...")
            self.matcher = product_matcher.ProductMatcher('data/catalog_optimized.csv', min_confidence=0.35)
            logger.info("✅ Product Matcher loaded")
            
            # Initialize Vibe Classifier
            logger.info("🎨 Loading Vibe Classifier...")
            self.classifier = vibe_classifier.VibeClassifier('data/vibeslist.json')
            logger.info("✅ Vibe Classifier loaded")
            
            logger.info("🎉 Pipeline initialization complete!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipeline: {e}")
            raise
    
    def _create_optimized_catalog(self):
        """Create an optimized catalog subset for faster processing"""
        catalog_path = 'data/catalog_optimized.csv'
        
        if os.path.exists(catalog_path):
            logger.info(f"📁 Using existing optimized catalog: {catalog_path}")
            return
        
        logger.info(f"📊 Creating optimized catalog with {self.catalog_size} products...")
        
        # Load full catalog
        full_catalog = pd.read_csv('data/images.csv')
        
        # Take a diverse subset of products (every nth product to get variety)
        step = len(full_catalog) // self.catalog_size
        if step < 1:
            step = 1
        
        optimized_catalog = full_catalog.iloc[::step].head(self.catalog_size)
        
        # Save optimized catalog
        optimized_catalog.to_csv(catalog_path, index=False)
        logger.info(f"✅ Created optimized catalog with {len(optimized_catalog)} products")
    
    def extract_keyframes(self, video_path: str, num_frames: int = 5) -> List[np.ndarray]:
        """Extract keyframes from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        # Extract frames at regular intervals
        if frame_count > 0:
            indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        
        cap.release()
        return frames
    
    def process_single_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a single video through the complete pipeline
        
        Args:
            video_path: Path to video file
            
        Returns:
            Results dictionary in required format
        """
        video_id = os.path.basename(video_path).replace('.mp4', '')
        logger.info(f"🎬 Processing video: {video_id}")
        
        start_time = time.time()
        
        try:
            # Extract keyframes
            logger.info("📹 Extracting keyframes...")
            frames = self.extract_keyframes(video_path, num_frames=5)
            if not frames:
                logger.warning(f"No frames extracted from {video_path}")
                return self._empty_result(video_id, "No frames extracted")
            
            logger.info(f"✅ Extracted {len(frames)} keyframes")
            
            # Detect fashion items
            logger.info("👗 Detecting fashion items...")
            all_detections = []
            for i, frame in enumerate(frames):
                detections = self.detector.detect_fashion_items(frame, frame_number=i)
                all_detections.extend(detections)
            
            logger.info(f"✅ Found {len(all_detections)} fashion detections")
            
            # Match products
            logger.info("🛍️ Matching products...")
            matched_products = []
            
            for detection in all_detections:
                # Get the frame where this detection was found
                frame_idx = detection.get('frame_number', 0)
                if frame_idx < len(frames):
                    frame = frames[frame_idx]
                    
                    # Extract detected region
                    bbox = detection['bbox']
                    x, y, w, h = [int(coord) for coord in bbox]
                    
                    # Ensure bbox is within frame bounds
                    h_frame, w_frame = frame.shape[:2]
                    x = max(0, min(x, w_frame - 1))
                    y = max(0, min(y, h_frame - 1))
                    w = max(1, min(w, w_frame - x))
                    h = max(1, min(h, h_frame - y))
                    
                    cropped_image = frame[y:y+h, x:x+w]
                    
                    if cropped_image.size > 0:
                        # Find matches
                        matches = self.matcher.find_matches(
                            cropped_image, 
                            detection.get('class_name', 'unknown'),
                            top_k=2
                        )
                        
                        for match in matches:
                            # Update match with detection info
                            match.update({
                                'type': detection.get('fashion_type', detection.get('class_name', 'unknown')),
                                'color': detection.get('color', 'unknown')
                            })
                            matched_products.append(match)
            
            logger.info(f"✅ Found {len(matched_products)} product matches")
            
            # Classify vibes
            logger.info("🎭 Classifying vibes...")
            vibes = self._classify_video_vibes(video_path)
            logger.info(f"✅ Classified vibes: {vibes}")
            
            # Create final result
            result = {
                "video_id": video_id,
                "vibes": vibes,
                "products": matched_products[:10]  # Limit to top 10 matches
            }
            
            processing_time = time.time() - start_time
            logger.info(f"🎉 Video processed successfully in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing video {video_id}: {e}")
            return self._empty_result(video_id, str(e))
    
    def _classify_video_vibes(self, video_path: str) -> List[str]:
        """Classify vibes for a video using available text data"""
        try:
            # Try to find associated text file
            text_file = video_path.replace('.mp4', '.txt')
            text_content = ""
            
            if os.path.exists(text_file):
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            
            # Also use video filename as context
            video_name = os.path.basename(video_path)
            combined_text = f"{text_content} {video_name}".strip()
            
            if combined_text:
                vibes = self.classifier.classify_vibes(combined_text, max_vibes=3)
                return vibes
            else:
                # If no text available, return empty list
                return []
                
        except Exception as e:
            logger.warning(f"Vibe classification failed: {e}")
            return []
    
    def _empty_result(self, video_id: str, error: str = None) -> Dict[str, Any]:
        """Create empty result structure"""
        result = {
            "video_id": video_id,
            "vibes": [],
            "products": []
        }
        if error:
            result["error"] = error
        return result
    
    def process_all_videos(self, video_dir: str = "data/videos") -> Dict[str, Any]:
        """
        Process all videos in the directory
        
        Args:
            video_dir: Directory containing video files
            
        Returns:
            Summary results
        """
        logger.info(f"🎬 Processing all videos in {video_dir}")
        
        # Find all MP4 files
        video_files = list(Path(video_dir).glob("*.mp4"))
        
        if not video_files:
            logger.error(f"No video files found in {video_dir}")
            return {"error": "No video files found", "results": []}
        
        logger.info(f"📁 Found {len(video_files)} videos to process")
        
        all_results = []
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        for i, video_file in enumerate(video_files, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"📹 Processing video {i}/{len(video_files)}: {video_file.name}")
            logger.info(f"{'='*50}")
            
            result = self.process_single_video(str(video_file))
            all_results.append(result)
            
            if "error" in result:
                failed += 1
            else:
                successful += 1
            
            # Progress update
            logger.info(f"✅ Progress: {i}/{len(video_files)} videos processed")
        
        total_time = time.time() - start_time
        
        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_videos": len(video_files),
            "successful": successful,
            "failed": failed,
            "processing_time_seconds": round(total_time, 2),
            "average_time_per_video": round(total_time / len(video_files), 2),
            "results": all_results
        }
        
        logger.info(f"\n🎉 PIPELINE COMPLETE!")
        logger.info(f"📊 Summary: {successful}/{len(video_files)} videos processed successfully")
        logger.info(f"⏱️ Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        logger.info(f"⚡ Average: {total_time/len(video_files):.2f}s per video")
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_file: str = "outputs/pipeline_results.json"):
        """Save results to JSON file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Results saved to: {output_file}")


def main():
    """Main entry point"""
    print("""
    🎬 FLICKD AI HACKATHON - FASHION PIPELINE 🎬
    ============================================
    Smart Tagging & Vibe Classification Engine
    ============================================
    """)
    
    try:
        # Initialize pipeline
        pipeline = FashionAIPipeline(catalog_size=1500)
        
        # Process all videos
        results = pipeline.process_all_videos()
        
        # Save results
        pipeline.save_results(results)
        
        # Display summary
        print(f"""
        🎉 PROCESSING COMPLETE! 🎉
        ========================
        ✅ Successful: {results['successful']} videos
        ❌ Failed: {results['failed']} videos
        ⏱️ Total time: {results['processing_time_seconds']}s
        ⚡ Average: {results['average_time_per_video']}s per video
        💾 Results saved to: outputs/pipeline_results.json
        
        Sample results:
        """)
        
        # Show sample results
        for i, result in enumerate(results['results'][:3]):
            print(f"📹 Video {i+1}: {result['video_id']}")
            print(f"   🎭 Vibes: {result['vibes']}")
            print(f"   🛍️ Products: {len(result['products'])} matches")
            if result['products']:
                print(f"   🏆 Top match: {result['products'][0]['matched_product_id']} (confidence: {result['products'][0]['confidence']:.2f})")
            print()
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main() 