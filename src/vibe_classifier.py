#!/usr/bin/env python3
"""
Vibe classification using NLP approaches
"""

import json
import logging
from typing import List, Dict, Any
import re

logger = logging.getLogger(__name__)

class VibeClassifier:
    """NLP-based vibe classifier for fashion videos"""
    
    def __init__(self, vibes_file_path: str):
        """
        Initialize vibe classifier
        
        Args:
            vibes_file_path: Path to JSON file containing vibe definitions
        """
        self.vibes = self._load_vibes(vibes_file_path)
        self.vibe_keywords = self._create_keyword_mapping()
        logger.info(f"Initialized VibeClassifier with {len(self.vibes)} vibes")
    
    def _load_vibes(self, vibes_file_path: str) -> List[str]:
        """Load vibe list from JSON file"""
        try:
            with open(vibes_file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'vibes' in data:
                    return data['vibes']
                else:
                    logger.warning(f"Unexpected format in {vibes_file_path}, using default vibes")
                    return self._get_default_vibes()
        except Exception as e:
            logger.error(f"Failed to load vibes from {vibes_file_path}: {e}")
            return self._get_default_vibes()
    
    def _get_default_vibes(self) -> List[str]:
        """Get default vibe list"""
        return [
            "Coquette", "Clean Girl", "Cottagecore", "Streetcore", 
            "Y2K", "Boho", "Party Glam"
        ]
    
    def _create_keyword_mapping(self) -> Dict[str, List[str]]:
        """Create keyword mapping for each vibe"""
        return {
            "Coquette": [
                "pink", "bow", "lace", "ribbon", "pearl", "cute", "sweet", 
                "feminine", "romantic", "soft", "delicate", "pretty", "girly",
                "coquette", "dainty", "flirty", "princess"
            ],
            "Clean Girl": [
                "minimal", "natural", "simple", "effortless", "fresh", 
                "dewy", "glowing", "no makeup", "slicked back", "clean",
                "cleangirl", "minimalist", "barely there", "nude", "natural glow",
                "fresh faced", "glowy", "simple", "easy", "breezy", "cotton",
                "linen", "comfortable", "relaxed", "casual", "everyday"
            ],
            "Cottagecore": [
                "floral", "vintage", "rustic", "cozy", "countryside", "garden", 
                "cottage", "pastoral", "prairie", "romantic", "ethereal",
                "cottagecore", "wildflower", "meadow", "cabin", "nature"
            ],
            "Streetcore": [
                "urban", "edgy", "street", "grunge", "cool", "alternative", 
                "rebellious", "bold", "tough", "underground", "streetcore",
                "hip hop", "sneakers", "oversized", "graphic", "distressed"
            ],
            "Y2K": [
                "2000s", "y2k", "metallic", "shiny", "futuristic", "cyber", 
                "tech", "digital", "chrome", "holographic", "rave",
                "neon", "butterfly", "platform", "low rise", "bedazzled"
            ],
            "Boho": [
                "bohemian", "free spirited", "flowing", "earthy", "natural", 
                "hippie", "artistic", "eclectic", "vintage", "layered",
                "boho", "fringe", "tassel", "paisley", "ethnic", "tribal",
                "maxi", "flowy", "festival", "desert", "wanderlust"
            ],
            "Party Glam": [
                "glam", "sparkle", "sequin", "glitter", "party", "night out", 
                "clubbing", "dressy", "elegant", "formal", "luxe",
                "party glam", "bling", "disco", "glamorous", "dramatic",
                "statement", "bold", "evening", "cocktail", "gala"
            ]
        }
    
    def classify_vibes(self, text: str, max_vibes: int = 3) -> List[str]:
        """
        Classify vibes based on text input
        
        Args:
            text: Input text (caption, description, hashtags)
            max_vibes: Maximum number of vibes to return
            
        Returns:
            List of classified vibes
        """
        if not text:
            return []
        
        # Normalize text
        text_lower = text.lower()
        
        # Calculate scores for each vibe
        vibe_scores = {}
        for vibe, keywords in self.vibe_keywords.items():
            score = 0
            for keyword in keywords:
                # Count keyword occurrences
                count = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower))
                score += count
                
                # Boost score for hashtags
                hashtag_pattern = r'#\w*' + re.escape(keyword.lower()) + r'\w*'
                hashtag_matches = len(re.findall(hashtag_pattern, text_lower))
                score += hashtag_matches * 2  # Give hashtags more weight
            
            if score > 0:
                vibe_scores[vibe] = score
        
        # Sort by score and return top vibes
        sorted_vibes = sorted(vibe_scores.items(), key=lambda x: x[1], reverse=True)
        return [vibe for vibe, score in sorted_vibes[:max_vibes]]
    
    def analyze_video_vibes(self, video_metadata: Dict[str, Any]) -> List[str]:
        """
        Analyze vibes for a video based on its metadata
        
        Args:
            video_metadata: Dictionary containing video metadata
            
        Returns:
            List of classified vibes
        """
        # Combine all text sources
        text_sources = []
        
        if 'caption' in video_metadata:
            text_sources.append(video_metadata['caption'])
        
        if 'description' in video_metadata:
            text_sources.append(video_metadata['description'])
        
        if 'hashtags' in video_metadata:
            if isinstance(video_metadata['hashtags'], list):
                text_sources.extend(video_metadata['hashtags'])
            else:
                text_sources.append(str(video_metadata['hashtags']))
        
        if 'title' in video_metadata:
            text_sources.append(video_metadata['title'])
        
        # Combine all text
        combined_text = ' '.join(text_sources)
        
        return self.classify_vibes(combined_text)
    
    def get_vibe_keywords(self, vibe: str) -> List[str]:
        """Get keywords for a specific vibe"""
        return self.vibe_keywords.get(vibe, [])
    
    def get_all_vibes(self) -> List[str]:
        """Get list of all available vibes"""
        return self.vibes.copy()