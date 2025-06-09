#!/usr/bin/env python3
"""
Test script to process a specific video and show detailed results
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Import our pipeline
from fashion_ai_pipeline import FashionAIPipeline

# Setup logging for detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_specific_video(video_name: str = "2025-05-28_13-40-09_UTC.mp4"):
    """Test processing on a specific video with detailed analysis"""
    
    video_path = f"data/videos/{video_name}"
    
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        return
    
    print(f"""
    🎬 TESTING SPECIFIC VIDEO 🎬
    ============================
    Video: {video_name}
    Path: {video_path}
    ============================
    """)
    
    try:
        # Initialize pipeline with smaller catalog for faster testing
        logger.info("🚀 Initializing pipeline...")
        pipeline = FashionAIPipeline(catalog_size=1000)
        
        # Check if text file exists
        text_file = video_path.replace('.mp4', '.txt')
        if os.path.exists(text_file):
            with open(text_file, 'r') as f:
                text_content = f.read()
            print(f"📄 Associated text content:")
            print(f"   {text_content.strip()}")
            print()
        
        # Process the video
        logger.info(f"🎬 Processing video: {video_name}")
        start_time = time.time()
        
        result = pipeline.process_single_video(video_path)
        
        processing_time = time.time() - start_time
        
        # Display detailed results
        print(f"""
        🎉 PROCESSING RESULTS 🎉
        ========================
        ⏱️ Processing time: {processing_time:.2f} seconds
        
        📊 SUMMARY:
        -----------
        🎭 Vibes found: {len(result['vibes'])}
        🛍️ Products matched: {len(result['products'])}
        """)
        
        # Show vibes
        if result['vibes']:
            print(f"🎭 VIBES CLASSIFIED:")
            for vibe in result['vibes']:
                print(f"   • {vibe}")
        else:
            print(f"🎭 VIBES: None detected")
        
        print()
        
        # Show products
        if result['products']:
            print(f"🛍️ PRODUCT MATCHES:")
            for i, product in enumerate(result['products'][:5], 1):  # Show top 5
                print(f"   {i}. {product['type']} ({product['color']})")
                print(f"      → Product ID: {product['matched_product_id']}")
                print(f"      → Match type: {product['match_type']}")
                print(f"      → Confidence: {product['confidence']:.3f}")
                print()
        else:
            print(f"🛍️ PRODUCTS: No matches found")
        
        # Save individual result
        output_file = f"outputs/test_{result['video_id']}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"💾 Detailed results saved to: {output_file}")
        
        # Show JSON output format
        print(f"""
        📋 JSON OUTPUT FORMAT:
        =====================
        {{
          "video_id": "{result['video_id']}",
          "vibes": {result['vibes']},
          "products": [
            // {len(result['products'])} product matches...
          ]
        }}
        """)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

def main():
    """Main function to run the test"""
    
    # Test different videos
    test_videos = [
        "2025-05-28_13-40-09_UTC.mp4",  # Different video from before
        "2025-05-27_13-46-16_UTC.mp4",  # Another different video
    ]
    
    for video in test_videos:
        print(f"\n{'='*60}")
        print(f"Testing video: {video}")
        print(f"{'='*60}")
        
        try:
            result = test_specific_video(video)
            
            if result and (result.get('vibes') or result.get('products')):
                print(f"✅ Success: Found {len(result.get('vibes', []))} vibes and {len(result.get('products', []))} products")
            else:
                print(f"⚠️ Processed but no results found")
                
        except Exception as e:
            print(f"❌ Failed to process {video}: {e}")
        
        print("\n" + "="*60)
        time.sleep(1)  # Brief pause between videos

if __name__ == "__main__":
    main() 