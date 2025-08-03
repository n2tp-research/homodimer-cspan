"""
Complete pipeline script for CSPAN homodimerization prediction.
This script runs all steps from feature extraction to evaluation.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import time
import yaml


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        sys.exit(1)
    else:
        print(f"SUCCESS: {description} completed in {elapsed_time:.1f} seconds")
        if result.stdout:
            print(f"Output:\n{result.stdout[-500:]}")  # Last 500 chars
    
    return result


def check_dependencies():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    
    try:
        import torch
        import transformers
        import datasets
        print("✓ Core packages installed")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ No CUDA available, will use CPU (slower)")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run complete CSPAN pipeline'
    )
    parser.add_argument(
        '--config', type=str, default='config.yml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU device ID'
    )
    parser.add_argument(
        '--skip-features', action='store_true',
        help='Skip feature extraction if already done'
    )
    parser.add_argument(
        '--skip-training', action='store_true',
        help='Skip training and use existing checkpoint'
    )
    parser.add_argument(
        '--checkpoint', type=str,
        help='Path to checkpoint (for skipping training)'
    )
    parser.add_argument(
        '--epochs', type=int,
        help='Override max epochs from config'
    )
    parser.add_argument(
        '--no-wandb', action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Load config to check paths
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    feature_cache = Path(config['data']['feature_cache_file'])
    checkpoint_dir = Path(config['experiment']['checkpoint_dir'])
    
    # Step 1: Feature Extraction
    if not args.skip_features:
        if feature_cache.exists():
            response = input(f"\nFeature cache {feature_cache} already exists. "
                           "Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Skipping feature extraction...")
                args.skip_features = True
    
    if not args.skip_features:
        cmd = [
            sys.executable, 'data/feature_extraction.py',
            '--config', args.config
        ]
        run_command(cmd, "Feature Extraction")
    else:
        print("\n✓ Using existing feature cache")
    
    # Step 2: Training
    if not args.skip_training:
        # Modify config if epochs specified
        if args.epochs:
            print(f"\nOverriding max_epochs to {args.epochs}")
            config['training']['max_epochs'] = args.epochs
            temp_config = 'config_temp.yml'
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)
            config_path = temp_config
        else:
            config_path = args.config
        
        cmd = [
            sys.executable, 'scripts/train.py',
            '--config', config_path,
            '--gpu', str(args.gpu)
        ]
        if args.no_wandb:
            cmd.append('--no-wandb')
        
        run_command(cmd, "Model Training")
        
        # Clean up temp config
        if args.epochs and os.path.exists(temp_config):
            os.remove(temp_config)
        
        # Find best checkpoint
        best_checkpoint = checkpoint_dir / 'best_model.pt'
        if not best_checkpoint.exists():
            print("ERROR: Training completed but no best model found!")
            sys.exit(1)
    else:
        if args.checkpoint:
            best_checkpoint = Path(args.checkpoint)
        else:
            best_checkpoint = checkpoint_dir / 'best_model.pt'
        
        if not best_checkpoint.exists():
            print(f"ERROR: Checkpoint {best_checkpoint} not found!")
            sys.exit(1)
        
        print(f"\n✓ Using existing checkpoint: {best_checkpoint}")
    
    # Step 3: Evaluation
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    cmd = [
        sys.executable, 'scripts/evaluate.py',
        '--checkpoint', str(best_checkpoint),
        '--config', args.config,
        '--output', str(results_dir)
    ]
    
    if torch.cuda.is_available():
        cmd.extend(['--device', 'cuda'])
    
    run_command(cmd, "Model Evaluation")
    
    # Step 4: Test Prediction
    print("\n" + "="*60)
    print("Testing Single Sequence Prediction")
    print("="*60)
    
    test_sequence = (
        "MKLLIAVGAGGIGQTTAAMLYDQLLQAGRGVVLVNARNPQGGYCPDECAIPKHVIQGEKYDV"
        "DAAMKAACGGINVDFIKEKDLDIILGEVITEGSILNKNSGKILMNAAEKYTSLLPDDVVEK"
    )
    
    cmd = [
        sys.executable, 'scripts/predict.py',
        '--checkpoint', str(best_checkpoint),
        '--config', args.config,
        '--sequence', test_sequence
    ]
    
    result = run_command(cmd, "Test Prediction")
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nResults saved to: {results_dir}/")
    print(f"Best model checkpoint: {best_checkpoint}")
    print(f"\nKey files generated:")
    print(f"  - Feature cache: {feature_cache}")
    print(f"  - Model checkpoint: {best_checkpoint}")
    print(f"  - Evaluation report: {results_dir}/evaluation_report.txt")
    print(f"  - Plots: {results_dir}/*.png")
    
    # Print final metrics if available
    report_path = results_dir / 'evaluation_report.txt'
    if report_path.exists():
        print("\nTest Set Performance Summary:")
        with open(report_path, 'r') as f:
            lines = f.readlines()
            in_overall = False
            for line in lines:
                if "Overall Performance:" in line:
                    in_overall = True
                elif in_overall and line.strip() == "":
                    break
                elif in_overall and ":" in line:
                    metric_line = line.strip()
                    if any(key in metric_line.lower() for key in 
                          ['auprc', 'mcc', 'f1', 'balanced_accuracy']):
                        print(f"  {metric_line}")


if __name__ == '__main__':
    main()