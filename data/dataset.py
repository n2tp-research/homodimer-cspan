"""
Data loading utilities for homodimer prediction using HuggingFace datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import yaml
import os


class HomodimerDataset(Dataset):
    """PyTorch dataset wrapper for the Synthyra/homodimer_benchmark dataset."""
    
    def __init__(
        self,
        split: str = "train",
        config_path: str = "config.yml",
        max_length: Optional[int] = None,
        filter_sequences: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the homodimer dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            config_path: Path to configuration file
            max_length: Maximum sequence length (None uses config value)
            filter_sequences: Whether to apply quality filters
            verbose: Print loading progress
        """
        self.split = split
        self.verbose = verbose
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.max_length = max_length or self.data_config['max_seq_length']
        self.min_length = self.data_config['min_seq_length']
        self.max_unknown_ratio = self.data_config['max_unknown_ratio']
        self.valid_amino_acids = set(self.data_config['valid_amino_acids'])
        
        # Load dataset from HuggingFace
        if self.verbose:
            print(f"Loading {split} split from Synthyra/homodimer_benchmark...")
        
        dataset = load_dataset(
            self.data_config['dataset_name'],
            split=split,
            cache_dir=self.data_config.get('cache_dir', None)
        )
        
        # Apply filtering if requested
        if filter_sequences:
            dataset = self._filter_sequences(dataset)
        
        # Convert to lists for easier access
        self.sequences = list(dataset['sequence'])
        self.labels = list(dataset['label'])
        
        # Calculate class weights if needed
        if self.data_config.get('compute_class_weights', True):
            self._compute_class_weights()
        
        if self.verbose:
            self._print_statistics()
    
    def _filter_sequences(self, dataset):
        """Apply quality filters to sequences."""
        if self.verbose:
            print("Applying sequence quality filters...")
            original_size = len(dataset)
        
        def is_valid_sequence(example):
            seq = example['sequence']
            seq_length = len(seq)
            
            # Check length constraints
            if seq_length < self.min_length or seq_length > self.max_length:
                return False
            
            # Check unknown residue ratio
            unknown_count = sum(1 for aa in seq if aa not in self.valid_amino_acids)
            unknown_ratio = unknown_count / seq_length if seq_length > 0 else 1.0
            
            return unknown_ratio < self.max_unknown_ratio
        
        dataset = dataset.filter(is_valid_sequence)
        
        if self.verbose:
            filtered_size = len(dataset)
            print(f"Filtered {original_size - filtered_size} sequences")
            print(f"Retained {filtered_size}/{original_size} sequences ({filtered_size/original_size:.1%})")
        
        return dataset
    
    def _compute_class_weights(self):
        """Compute class weights for handling imbalance."""
        labels = np.array(self.labels)
        n_samples = len(labels)
        n_classes = 2
        
        class_counts = np.bincount(labels)
        self.class_weights = n_samples / (n_classes * class_counts)
        
        if self.verbose:
            print(f"Class weights: {self.class_weights}")
    
    def _print_statistics(self):
        """Print dataset statistics."""
        labels = np.array(self.labels)
        seq_lengths = [len(seq) for seq in self.sequences]
        
        print(f"\n{self.split.upper()} Dataset Statistics:")
        print(f"Total sequences: {len(self)}")
        print(f"Positive samples: {np.sum(labels)} ({np.mean(labels):.1%})")
        print(f"Negative samples: {len(labels) - np.sum(labels)} ({1 - np.mean(labels):.1%})")
        print(f"Sequence length - Mean: {np.mean(seq_lengths):.0f}, Std: {np.std(seq_lengths):.0f}")
        print(f"Sequence length - Min: {np.min(seq_lengths)}, Max: {np.max(seq_lengths)}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        return {
            'sequence': self.sequences[idx],
            'label': self.labels[idx],
            'index': idx
        }
    
    def get_dataloader(
        self,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        num_workers: Optional[int] = None,
        **kwargs
    ) -> DataLoader:
        """Create a DataLoader for this dataset."""
        # Use config defaults if not specified
        if batch_size is None:
            batch_size = self.config['training']['batch_size']
        if shuffle is None:
            shuffle = (self.split == 'train')
        if num_workers is None:
            num_workers = self.data_config['num_workers']
        
        # Default DataLoader settings from config
        loader_kwargs = {
            'pin_memory': self.data_config.get('pin_memory', True),
            'prefetch_factor': self.data_config.get('prefetch_factor', 2),
            'persistent_workers': self.data_config.get('persistent_workers', True) and num_workers > 0
        }
        loader_kwargs.update(kwargs)
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            **loader_kwargs
        )
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching sequences."""
        sequences = [item['sequence'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        indices = torch.tensor([item['index'] for item in batch], dtype=torch.long)
        
        return {
            'sequences': sequences,  # Keep as list for ESM2 processing
            'labels': labels,
            'indices': indices
        }


class SequenceBucketSampler:
    """Sampler that groups sequences by length to minimize padding."""
    
    def __init__(
        self,
        dataset: HomodimerDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Sort sequences by length
        self.length_indices = sorted(
            range(len(dataset)),
            key=lambda i: len(dataset.sequences[i])
        )
        
        # Create buckets
        self.buckets = []
        for i in range(0, len(self.length_indices), batch_size):
            bucket = self.length_indices[i:i + batch_size]
            if len(bucket) == batch_size or not drop_last:
                self.buckets.append(bucket)
    
    def __iter__(self):
        # Shuffle buckets if requested
        if self.shuffle:
            indices = torch.randperm(len(self.buckets))
            buckets = [self.buckets[i] for i in indices]
        else:
            buckets = self.buckets
        
        # Yield all indices
        for bucket in buckets:
            if self.shuffle:
                # Shuffle within bucket
                perm = torch.randperm(len(bucket))
                bucket = [bucket[i] for i in perm]
            
            for idx in bucket:
                yield idx
    
    def __len__(self):
        if self.drop_last:
            return len(self.buckets) * self.batch_size
        else:
            return len(self.dataset)


def get_dataloaders(
    config_path: str = "config.yml",
    batch_size: Optional[int] = None,
    use_bucket_sampling: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config_path: Path to configuration file
        batch_size: Batch size (uses config default if None)
        use_bucket_sampling: Whether to use length-based bucket sampling
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = HomodimerDataset('train', config_path)
    val_dataset = HomodimerDataset('validation', config_path)
    test_dataset = HomodimerDataset('test', config_path)
    
    # Get batch size from config if not specified
    if batch_size is None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        batch_size = config['training']['batch_size']
    
    # Create dataloaders
    if use_bucket_sampling:
        # Use bucket sampling for training
        train_sampler = SequenceBucketSampler(
            train_dataset, batch_size, shuffle=True, drop_last=True
        )
        train_loader = train_dataset.get_dataloader(
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False  # Sampler handles shuffling
        )
    else:
        train_loader = train_dataset.get_dataloader(
            batch_size=batch_size,
            shuffle=True
        )
    
    # Regular loading for validation and test
    val_loader = val_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = test_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    
    # Create a simple test config if it doesn't exist
    if not os.path.exists("config.yml"):
        test_config = {
            'data': {
                'dataset_name': 'Synthyra/homodimer_benchmark',
                'min_seq_length': 50,
                'max_seq_length': 5000,
                'max_unknown_ratio': 0.05,
                'valid_amino_acids': 'ACDEFGHIKLMNPQRSTVWY',
                'compute_class_weights': True,
                'num_workers': 2,
                'pin_memory': True,
                'prefetch_factor': 2,
                'persistent_workers': True
            },
            'training': {
                'batch_size': 16
            }
        }
        with open("config.yml", 'w') as f:
            yaml.dump(test_config, f)
    
    # Test dataset loading
    dataset = HomodimerDataset('train', verbose=True)
    print(f"\nFirst sample: {dataset[0]}")
    
    # Test dataloader
    loader = dataset.get_dataloader(batch_size=4)
    batch = next(iter(loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch shapes: labels={batch['labels'].shape}, indices={batch['indices'].shape}")
    print(f"First sequence in batch: {batch['sequences'][0][:50]}...")