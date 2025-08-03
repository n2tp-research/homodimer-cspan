#!/usr/bin/env python3
"""
Precompute and cache all ESM2 features before training.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.feature_extraction import precompute_all_features


def main():
    parser = argparse.ArgumentParser(description='Precompute ESM2 features for all dataset splits')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to configuration file')
    parser.add_argument('--force', action='store_true',
                       help='Force recomputation even if cache exists')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Precomputing ESM2 features for homodimer dataset")
    print("="*80)
    
    precompute_all_features(
        config_path=args.config,
        force_recompute=args.force
    )
    
    print("\n" + "="*80)
    print("Feature precomputation complete!")
    print("You can now run training with cached features.")
    print("="*80)


if __name__ == "__main__":
    main()