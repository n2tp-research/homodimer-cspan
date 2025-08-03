#!/usr/bin/env python3
"""
Fast feature extraction using single GPU with batch processing and multiprocessing.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import h5py
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import HomodimerDataset
from data.feature_extraction import ESM2FeatureExtractor


class FastFeatureExtractor:
    """Fast feature extraction with batching and caching."""
    
    def __init__(self, config_path: str, batch_size: int = 32):
        self.config_path = config_path
        self.batch_size = batch_size
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize extractor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = ESM2FeatureExtractor(
            config_path=config_path,
            device=device,
            use_half_precision=True,
            verbose=False
        )
        
        # Get cache info
        self.cache_path = self.config['data']['feature_cache_file']
        self.cache_lock = threading.Lock()
        
    def check_cache_batch(self, indices: list, split: str) -> tuple:
        """Check which indices are already cached."""
        cached = []
        uncached = []
        
        with h5py.File(self.cache_path, 'r') as cache:
            for idx in indices:
                cache_key = f"{split}/{idx}"
                if cache_key in cache:
                    cached.append(idx)
                else:
                    uncached.append(idx)
        
        return cached, uncached
    
    def process_dataset(self, split: str):
        """Process entire dataset with efficient batching."""
        print(f"\nProcessing {split} split...")
        
        # Load dataset
        dataset = HomodimerDataset(split=split, config_path=self.config_path, verbose=True)
        total_sequences = len(dataset)
        
        # First pass: check what's already cached
        print("Checking cache...")
        all_indices = list(range(total_sequences))
        cached_indices = set()
        uncached_indices = []
        
        # Check cache in chunks
        chunk_size = 1000
        for i in tqdm(range(0, total_sequences, chunk_size), desc="Cache check"):
            chunk = all_indices[i:i+chunk_size]
            try:
                cached, uncached = self.check_cache_batch(chunk, split)
                cached_indices.update(cached)
                uncached_indices.extend(uncached)
            except:
                # If cache doesn't exist, all are uncached
                uncached_indices.extend(chunk)
        
        print(f"Found {len(cached_indices)} cached, {len(uncached_indices)} to process")
        
        if not uncached_indices:
            print(f"{split} split already fully cached!")
            return
        
        # Process uncached sequences in large batches
        sequences_to_process = [dataset.sequences[i] for i in uncached_indices]
        
        # Process in batches
        pbar = tqdm(total=len(uncached_indices), desc=f"Extracting {split}")
        
        for i in range(0, len(uncached_indices), self.batch_size):
            batch_indices = uncached_indices[i:i+self.batch_size]
            batch_sequences = sequences_to_process[i:i+self.batch_size]
            
            # Extract features
            try:
                features = self.extractor.extract_features(
                    batch_sequences,
                    split=split,
                    indices=batch_indices,
                    use_cache=False,
                    save_to_cache=True
                )
                pbar.update(len(batch_sequences))
            except Exception as e:
                print(f"\nError processing batch: {e}")
                # Try smaller batch size
                for j in range(len(batch_sequences)):
                    try:
                        features = self.extractor.extract_features(
                            [batch_sequences[j]],
                            split=split,
                            indices=[batch_indices[j]],
                            use_cache=False,
                            save_to_cache=True
                        )
                        pbar.update(1)
                    except:
                        print(f"Skipping sequence {batch_indices[j]}")
                        pbar.update(1)
        
        pbar.close()
        print(f"Completed {split} split!")


def main():
    parser = argparse.ArgumentParser(description='Fast ESM2 feature extraction')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to configuration file')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                       help='Dataset splits to process')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Fast ESM2 Feature Extraction (Single GPU)")
    print("="*80)
    print(f"Batch size: {args.batch_size}")
    print(f"Splits: {args.splits}")
    
    # Create extractor
    extractor = FastFeatureExtractor(args.config, args.batch_size)
    
    # Process each split
    for split in args.splits:
        extractor.process_dataset(split)
    
    # Cleanup
    extractor.extractor.close()
    
    print("\n" + "="*80)
    print("Feature extraction complete!")
    print("="*80)


if __name__ == "__main__":
    main()