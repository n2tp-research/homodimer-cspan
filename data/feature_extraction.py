"""
ESM2 feature extraction with efficient caching.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import h5py
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
import os
import yaml
from pathlib import Path
import hashlib
import gc


class ESM2FeatureExtractor:
    """Extract and cache ESM2 embeddings for protein sequences."""
    
    def __init__(
        self,
        config_path: str = "config.yml",
        cache_path: Optional[str] = None,
        device: Optional[str] = None,
        use_half_precision: bool = True,
        verbose: bool = True
    ):
        """
        Initialize ESM2 feature extractor.
        
        Args:
            config_path: Path to configuration file
            cache_path: Path to cache file (uses config default if None)
            device: Device to use ('cuda' or 'cpu', auto-detect if None)
            use_half_precision: Use FP16 for memory efficiency
            verbose: Print progress information
        """
        self.verbose = verbose
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.model_config = self.config['model']
        
        # Set up paths
        self.cache_path = cache_path or self.data_config['feature_cache_file']
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if self.verbose:
            print(f"Using device: {self.device}")
        
        # Load ESM2 model and tokenizer
        self.model_name = self.model_config['esm_model']
        self.use_half = use_half_precision and self.device.type == 'cuda'
        
        self._load_model()
        
        # Feature extraction settings
        self.max_length = self.data_config['esm_max_length']
        self.batch_size = self.data_config['esm_batch_size']
        
        # Initialize cache
        self.cache = self._initialize_cache()
    
    def _load_model(self):
        """Load ESM2 model and tokenizer."""
        if self.verbose:
            print(f"Loading ESM2 model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Move to device and set precision
        self.model = self.model.to(self.device)
        if self.use_half:
            self.model = self.model.half()
        
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        
        if self.verbose:
            print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def _initialize_cache(self) -> Optional[h5py.File]:
        """Initialize or load existing cache."""
        if not self.data_config.get('use_feature_cache', True):
            return None
        
        # Check if cache exists
        if os.path.exists(self.cache_path):
            if self.verbose:
                print(f"Loading existing cache from {self.cache_path}")
            try:
                cache = h5py.File(self.cache_path, 'a')
                # Verify cache metadata
                if 'metadata' in cache.attrs:
                    cached_model = cache.attrs.get('model_name', '')
                    if cached_model != self.model_name:
                        print(f"Warning: Cache was created with {cached_model}, but using {self.model_name}")
                return cache
            except Exception as e:
                print(f"Error loading cache: {e}. Creating new cache.")
                os.remove(self.cache_path)
        
        # Create new cache
        if self.verbose:
            print(f"Creating new cache at {self.cache_path}")
        
        cache = h5py.File(self.cache_path, 'w')
        cache.attrs['model_name'] = self.model_name
        cache.attrs['embedding_dim'] = self.embedding_dim
        cache.attrs['max_length'] = self.max_length
        
        # Create groups for each dataset split
        for split in ['train', 'valid', 'test']:
            cache.create_group(split)
        
        return cache
    
    def _get_sequence_hash(self, sequence: str) -> str:
        """Generate a hash for a sequence for cache lookup."""
        return hashlib.md5(sequence.encode()).hexdigest()
    
    def _extract_single_batch(
        self,
        sequences: List[str]
    ) -> List[torch.Tensor]:
        """Extract features for a single batch of sequences."""
        # Tokenize sequences
        inputs = self.tokenizer(
            sequences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        
        # Process each sequence
        features = []
        for i, mask in enumerate(inputs['attention_mask']):
            # Get actual sequence length (excluding special tokens)
            seq_len = mask.sum().item() - 2  # Exclude [CLS] and [SEP]
            
            # Extract only real residue embeddings
            seq_embedding = embeddings[i, 1:seq_len+1, :]  # Skip [CLS], stop before [SEP]
            
            # Convert to CPU and float32 for storage
            seq_embedding = seq_embedding.cpu().float()
            features.append(seq_embedding)
        
        return features
    
    def extract_features(
        self,
        sequences: List[str],
        split: str = 'train',
        indices: Optional[List[int]] = None,
        use_cache: bool = True,
        save_to_cache: bool = True
    ) -> List[np.ndarray]:
        """
        Extract ESM2 features for a list of sequences.
        
        Args:
            sequences: List of protein sequences
            split: Dataset split for caching ('train', 'validation', 'test')
            indices: Optional indices for cache storage
            use_cache: Whether to use cached features if available
            save_to_cache: Whether to save extracted features to cache
        
        Returns:
            List of feature arrays, each of shape (seq_len, embedding_dim)
        """
        features = []
        sequences_to_extract = []
        extraction_indices = []
        
        # Check cache for existing features
        if use_cache and self.cache is not None and indices is not None:
            for i, (seq, idx) in enumerate(zip(sequences, indices)):
                cache_key = f"{split}/{idx}"
                
                if cache_key in self.cache:
                    # Load from cache
                    cached_features = self.cache[cache_key][()]
                    features.append(cached_features)
                else:
                    # Mark for extraction
                    sequences_to_extract.append(seq)
                    extraction_indices.append((i, idx))
        else:
            sequences_to_extract = sequences
            extraction_indices = [(i, indices[i] if indices else i) 
                                for i in range(len(sequences))]
        
        # Extract features for uncached sequences
        if sequences_to_extract:
            if self.verbose:
                print(f"Extracting features for {len(sequences_to_extract)} sequences...")
            
            extracted_features = []
            
            # Process in batches
            for i in tqdm(range(0, len(sequences_to_extract), self.batch_size),
                         disable=not self.verbose):
                batch = sequences_to_extract[i:i + self.batch_size]
                batch_features = self._extract_single_batch(batch)
                extracted_features.extend(batch_features)
            
            # Save to cache if requested
            if save_to_cache and self.cache is not None and indices is not None:
                for (orig_idx, cache_idx), feat in zip(extraction_indices, extracted_features):
                    cache_key = f"{split}/{cache_idx}"
                    
                    # Convert to numpy for storage
                    feat_np = feat.numpy()
                    
                    # Create dataset with compression
                    self.cache.create_dataset(
                        cache_key,
                        data=feat_np,
                        compression=self.data_config.get('compression', 'gzip'),
                        compression_opts=self.data_config.get('compression_level', 4)
                    )
                
                # Flush cache to disk
                self.cache.flush()
            
            # Insert extracted features at correct positions
            if use_cache and self.cache is not None:
                # Create full feature list with cached and extracted features
                full_features = [None] * len(sequences)
                
                # Fill in cached features
                cache_idx = 0
                extract_idx = 0
                
                for i in range(len(sequences)):
                    if cache_idx < len(features) and extract_idx < len(extraction_indices):
                        if extraction_indices[extract_idx][0] == i:
                            full_features[i] = extracted_features[extract_idx].numpy()
                            extract_idx += 1
                        else:
                            full_features[i] = features[cache_idx]
                            cache_idx += 1
                    elif cache_idx < len(features):
                        full_features[i] = features[cache_idx]
                        cache_idx += 1
                    else:
                        full_features[i] = extracted_features[extract_idx].numpy()
                        extract_idx += 1
                
                features = full_features
            else:
                features = [feat.numpy() for feat in extracted_features]
        
        return features
    
    def extract_dataset_features(
        self,
        dataset,
        split: str,
        force_recompute: bool = False
    ):
        """
        Extract features for an entire dataset split.
        
        Args:
            dataset: HomodimerDataset instance
            split: Dataset split name
            force_recompute: Force recomputation even if cached
        """
        if self.verbose:
            print(f"\nExtracting features for {split} split...")
        
        # Get all sequences and indices
        sequences = dataset.sequences
        indices = list(range(len(dataset)))
        
        # Extract features
        features = self.extract_features(
            sequences,
            split=split,
            indices=indices,
            use_cache=not force_recompute,
            save_to_cache=True
        )
        
        if self.verbose:
            print(f"Completed feature extraction for {len(features)} sequences")
        
        return features
    
    def get_collate_fn(self):
        """
        Get a collate function that extracts features on-the-fly.
        
        Returns:
            Collate function for DataLoader
        """
        # Create a simple collate function that doesn't use CUDA
        # Features will be extracted in the main process
        def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
            sequences = [item['sequence'] for item in batch]
            labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
            indices = [item['index'] for item in batch]
            
            # Don't extract features here if using multiple workers
            # Return sequences to be processed in main thread
            return {
                'sequences': sequences,
                'labels': labels,
                'indices': torch.tensor(indices, dtype=torch.long),
                'needs_features': True
            }
            
            # Pad features to same length
            max_len = max(feat.shape[0] for feat in features)
            padded_features = []
            attention_masks = []
            
            for feat in features:
                seq_len = feat.shape[0]
                if seq_len < max_len:
                    # Pad with zeros
                    padding = np.zeros((max_len - seq_len, feat.shape[1]))
                    feat = np.concatenate([feat, padding], axis=0)
                
                padded_features.append(feat)
                
                # Create attention mask
                mask = np.ones(max_len)
                mask[seq_len:] = 0
                attention_masks.append(mask)
            
            # Convert to tensors
            features_tensor = torch.tensor(np.stack(padded_features), dtype=torch.float32)
            attention_mask = torch.tensor(np.stack(attention_masks), dtype=torch.float32)
            
            return {
                'features': features_tensor,
                'attention_mask': attention_mask,
                'labels': labels,
                'indices': torch.tensor(indices, dtype=torch.long)
            }
        
        return collate_fn
    
    def close(self):
        """Close the cache file."""
        if self.cache is not None:
            self.cache.close()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()


def precompute_all_features(
    config_path: str = "config.yml",
    force_recompute: bool = False
):
    """
    Precompute and cache all ESM2 features for the dataset.
    
    Args:
        config_path: Path to configuration file
        force_recompute: Force recomputation even if cache exists
    """
    from data.dataset import HomodimerDataset
    
    # Initialize feature extractor
    extractor = ESM2FeatureExtractor(config_path=config_path, verbose=True)
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        print(f"\n{'='*50}")
        print(f"Processing {split} split")
        print('='*50)
        
        # Load dataset
        dataset = HomodimerDataset(split=split, config_path=config_path, verbose=True)
        
        # Extract features
        extractor.extract_dataset_features(
            dataset,
            split=split,
            force_recompute=force_recompute
        )
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Close extractor
    extractor.close()
    
    print("\nFeature extraction complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Precompute ESM2 features")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of all features"
    )
    
    args = parser.parse_args()
    
    precompute_all_features(
        config_path=args.config,
        force_recompute=args.force
    )