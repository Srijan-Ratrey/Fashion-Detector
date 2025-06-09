#!/usr/bin/env python3
"""
Main pipeline for Fashion AI Video Analysis
"""

import argparse
import logging
import sys
from pathlib import Path
import json

# Import our modules
from .utils import setup_logging, extract_video_id, save_json_output, load_vibes_list, get_video_files
from .video_processor import VideoProcessor
from .fashion_detector import FashionDetector
from .vibe_classifier import VibeClassifier


def process_single_video(video_path: str, output_dir: str = "outputs") -> dict:
    """
    Process a single video and return results
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for results
        
    Returns:
        Results dictionary
    """
    logger = logging.getLogger(__name__)
    video_id = extract_video_id(video_path)
    
    logger.info(f"Processing video: {video_id}")
    
    try:
        # Initialize components
        video_processor = VideoProcessor(fps=2, max_frames=10)
        fashion_detector = FashionDetector()
        vibe_classifier = VibeClassifier()
        
        # Extract frames
        frames = video_processor.extract_keyframes_uniform(video_path, num_frames=5)
        logger.info(f"Extracted {len(frames)} frames")
        
        # Detect fashion items
        all_detections = fashion_detector.detect_batch(frames)
        
        # Classify vibes (simple rule-based for now)
        vibes = vibe_classifier.classify_video_vibes(video_path)
        
        # Create output structure
        results = {
            "video_id": video_id,
            "vibes": vibes,
            "products": []
        }
        
        # For now, add mock products to demonstrate structure
        if any(all_detections):
            # Mock product matching results
            results["products"] = [
                {
                    "type": "top",
                    "color": "white",
                    "matched_product_id": "14976",
                    "match_type": "similar",
                    "confidence": 0.85
                }
            ]
        
        # Save results
        output_path = f"{output_dir}/{video_id}.json"
        save_json_output(results, output_path)
        
        logger.info(f"Saved results to {output_path}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to process {video_path}: {e}")
        return {"video_id": video_id, "vibes": [], "products": [], "error": str(e)}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Fashion AI Video Analysis Pipeline")
    parser.add_argument("--video", help="Single video file to process")
    parser.add_argument("--input", help="Directory containing videos")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    logger.info("Starting Fashion AI Video Analysis")
    
    # Determine input videos
    video_files = []
    if args.video:
        video_files = [args.video]
    elif args.input:
        video_files = get_video_files(args.input)
    else:
        # Default to data/videos if it exists
        if Path("data/videos").exists():
            video_files = get_video_files("data/videos")
        else:
            logger.error("No input specified. Use --video or --input")
            sys.exit(1)
    
    if not video_files:
        logger.error("No video files found")
        sys.exit(1)
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    # Process videos
    all_results = []
    for video_file in video_files:
        try:
            result = process_single_video(video_file, args.output)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {video_file}: {e}")
    
    logger.info(f"Processed {len(all_results)} videos successfully")
    
    # Save summary
    summary = {
        "total_videos": len(video_files),
        "successful": len([r for r in all_results if "error" not in r]),
        "failed": len([r for r in all_results if "error" in r]),
        "results": all_results
    }
    
    save_json_output(summary, f"{args.output}/processing_summary.json")
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
