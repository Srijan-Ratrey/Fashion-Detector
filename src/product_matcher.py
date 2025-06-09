#!/usr/bin/env python3
"""
Product matching with CLIP + FAISS
"""

import torch
import clip
import numpy as np
from PIL import Image
import faiss
import logging
import pandas as pd
import requests
import cv2
from typing import List, Dict, Optional, Union
from pathlib import Path
import os
import tempfile

logger = logging.getLogger(__name__)

class ProductMatcher:
    def __init__(
        self,
        catalog_csv_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        min_confidence: float = 0.4,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize product matcher with CLIP and FAISS
        
        Args:
            catalog_csv_path: Path to CSV file containing product catalog
            device: Device to run model on ('cuda' or 'cpu')
            min_confidence: Minimum similarity threshold
            cache_dir: Directory to cache embeddings
        """
        self.device = device
        self.min_confidence = min_confidence
        self.cache_dir = cache_dir or "cache/embeddings"
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load catalog from CSV
        self.catalog_df = pd.read_csv(catalog_csv_path)

        # Handle different possible column names
        image_url_col = None
        product_id_col = None

        # Check for various column name formats
        for col in self.catalog_df.columns:
            col_lower = col.lower()
            if 'image' in col_lower and 'url' in col_lower:
                image_url_col = col
            elif 'id' in col_lower:
                product_id_col = col

        # Fallback to exact column names
        if image_url_col is None:
            if 'image_url' in self.catalog_df.columns:
                image_url_col = 'image_url'
            elif 'Image URL' in self.catalog_df.columns:
                image_url_col = 'Image URL'

        if product_id_col is None:
            if 'id' in self.catalog_df.columns:
                product_id_col = 'id'
            elif 'Product ID' in self.catalog_df.columns:
                product_id_col = 'Product ID'

        self.catalog_images = self.catalog_df[image_url_col].tolist() if image_url_col else []
        self.catalog_ids = self.catalog_df[product_id_col].tolist() if product_id_col else []

        logger.info(f"Loaded catalog with {len(self.catalog_images)} products using columns: {image_url_col}, {product_id_col}")
        
        # Load CLIP model
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=device)
            logger.info(f"Loaded CLIP model on {device}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
        
        # Compute or load embeddings
        self.embeddings = self._load_or_compute_embeddings()
        
        # Initialize FAISS index
        if self.embeddings.size > 0:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings.astype(np.float32))
            logger.info(f"Initialized FAISS index with {len(self.catalog_images)} catalog items")
        else:
            self.index = None
            logger.warning("No embeddings computed, index not initialized")

    def _download_image(self, url: str) -> Optional[str]:
        """Download image from URL to temporary file"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                f.write(response.content)
                return f.name
                
        except Exception as e:
            logger.warning(f"Failed to download image from {url}: {e}")
            return None

    def _get_cache_path(self, image_url: str) -> str:
        """Get path for cached embedding"""
        # Create filename from URL hash
        import hashlib
        url_hash = hashlib.md5(image_url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.npy")

    def _load_or_compute_embeddings(self) -> np.ndarray:
        """Load cached embeddings or compute new ones"""
        embs = []
        
        for i, img_url in enumerate(self.catalog_images):
            cache_path = self._get_cache_path(img_url)
            
            # Try to load from cache
            if os.path.exists(cache_path):
                try:
                    emb = np.load(cache_path)
                    embs.append(emb.reshape(1, -1))
                    logger.debug(f"Loaded cached embedding for product {i}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding for {img_url}: {e}")
            
            # Download and compute new embedding
            temp_path = self._download_image(img_url)
            if temp_path:
                try:
                    img = Image.open(temp_path).convert('RGB')
                    img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        emb = self.model.encode_image(img_tensor)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                        emb_np = emb.cpu().numpy()
                    
                    # Cache embedding
                    np.save(cache_path, emb_np)
                    embs.append(emb_np)
                    logger.debug(f"Computed new embedding for product {i}")
                    
                    # Clean up temp file
                    os.unlink(temp_path)
                    
                except Exception as e:
                    logger.error(f"Failed to process image {img_url}: {e}")
                    # Add zero embedding as placeholder
                    embs.append(np.zeros((1, 512)))
                    
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
            else:
                # Add zero embedding for failed downloads
                embs.append(np.zeros((1, 512)))
        
        if embs:
            return np.vstack(embs)
        else:
            return np.array([]).reshape(0, 512)

    def find_matches(
        self,
        query_image: np.ndarray,
        item_type: str = None,
        top_k: int = 3
    ) -> List[Dict[str, any]]:
        """
        Find matches for a query image (numpy array)
        
        Args:
            query_image: Query image as numpy array (OpenCV format)
            item_type: Type of item (optional, for filtering)
            top_k: Number of top matches to return
            
        Returns:
            List of matches with similarity scores
        """
        if self.index is None:
            logger.warning("Index not initialized, no matches possible")
            return []
        
        try:
            # Convert OpenCV image (BGR) to PIL Image (RGB)
            if len(query_image.shape) == 3:
                query_image_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
            else:
                query_image_rgb = query_image
            
            pil_image = Image.fromarray(query_image_rgb)
            
            # Preprocess and encode
            img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Get query embedding
            with torch.no_grad():
                query_emb = self.model.encode_image(img_tensor)
                query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
            
            # Search FAISS index
            D, I = self.index.search(
                query_emb.cpu().numpy().astype(np.float32),
                min(top_k, len(self.catalog_ids))
            )
            
            # Format results
            results = []
            for idx, score in zip(I[0], D[0]):
                if float(score) >= self.min_confidence:
                    # Get product info
                    product_id = self.catalog_ids[idx]
                    product_row = self.catalog_df.iloc[idx]
                    
                    match_type = "exact" if score > 0.9 else "similar"
                    
                    result = {
                        "type": item_type or "unknown",
                        "color": "unknown",  # Could be enhanced with color detection
                        "matched_product_id": product_id,
                        "match_type": match_type,
                        "confidence": float(score)
                    }
                    
                    # Add product details if available
                    if 'Name' in product_row:
                        result["product_name"] = product_row['Name']
                    
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Product matching failed: {e}")
            return []

    def match(
        self,
        query_img_path: str,
        top_k: int = 3,
        batch_size: int = 1
    ) -> List[Dict[str, float]]:
        """
        Match query image against catalog (legacy method)
        
        Args:
            query_img_path: Path to query image
            top_k: Number of top matches to return
            batch_size: Batch size for processing
            
        Returns:
            List of matches with similarity scores
        """
        try:
            # Load and preprocess image
            img = self.preprocess(Image.open(query_img_path)).unsqueeze(0).to(self.device)
            
            # Get query embedding
            with torch.no_grad():
                query_emb = self.model.encode_image(img)
                query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
            
            # Search FAISS index
            D, I = self.index.search(
                query_emb.cpu().numpy().astype(np.float32),
                min(top_k, len(self.catalog_ids))
            )
            
            # Format results
            results = []
            for idx, score in zip(I[0], D[0]):
                if float(score) >= self.min_confidence:
                    results.append({
                        "matched_product_id": self.catalog_ids[idx],
                        "similarity": float(score)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Product matching failed for {query_img_path}: {e}")
            return []

    def match_batch(
        self,
        query_img_paths: List[str],
        top_k: int = 3,
        batch_size: int = 4
    ) -> List[List[Dict[str, float]]]:
        """
        Match multiple query images against catalog
        
        Args:
            query_img_paths: List of paths to query images
            top_k: Number of top matches to return per image
            batch_size: Batch size for processing
            
        Returns:
            List of match results for each query image
        """
        all_results = []
        
        # Process in batches
        for i in range(0, len(query_img_paths), batch_size):
            batch_paths = query_img_paths[i:i + batch_size]
            
            try:
                # Load and preprocess batch
                batch_imgs = []
                for path in batch_paths:
                    img = self.preprocess(Image.open(path)).unsqueeze(0)
                    batch_imgs.append(img)
                
                batch = torch.cat(batch_imgs).to(self.device)
                
                # Get batch embeddings
                with torch.no_grad():
                    batch_embs = self.model.encode_image(batch)
                    batch_embs = batch_embs / batch_embs.norm(dim=-1, keepdim=True)
                
                # Search FAISS index
                D, I = self.index.search(
                    batch_embs.cpu().numpy().astype(np.float32),
                    min(top_k, len(self.catalog_ids))
                )
                
                # Format results for each image in batch
                for j in range(len(batch_paths)):
                    results = []
                    for idx, score in zip(I[j], D[j]):
                        if float(score) >= self.min_confidence:
                            results.append({
                                "matched_product_id": self.catalog_ids[idx],
                                "similarity": float(score)
                            })
                    all_results.append(results)
                
            except Exception as e:
                logger.error(f"Batch matching failed for batch {i//batch_size}: {e}")
                # Add empty results for failed batch
                all_results.extend([[] for _ in range(len(batch_paths))])
        
        return all_results
