"""
Test script to verify CSPAN installation and basic functionality.
"""

import sys
import torch
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    modules = [
        ('torch', torch.__version__),
        ('transformers', None),
        ('datasets', None),
        ('numpy', np.__version__),
        ('yaml', None),
        ('h5py', None),
        ('sklearn', None),
        ('tqdm', None),
        ('einops', None),
    ]
    
    all_ok = True
    for module_name, version in modules:
        try:
            if version:
                print(f"✓ {module_name}: {version}")
            else:
                module = __import__(module_name)
                version = getattr(module, '__version__', 'installed')
                print(f"✓ {module_name}: {version}")
        except ImportError:
            print(f"✗ {module_name}: NOT INSTALLED")
            all_ok = False
    
    return all_ok


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠ CUDA not available (CPU mode will be slower)")
        return False


def test_project_structure():
    """Test if project structure is correct."""
    print("\nTesting project structure...")
    
    required_dirs = ['data', 'models', 'utils', 'scripts', 'configs']
    required_files = [
        'config.yml',
        'data/dataset.py',
        'data/feature_extraction.py',
        'models/cspan.py',
        'models/layers.py',
        'utils/losses.py',
        'utils/metrics.py',
        'scripts/train.py',
        'scripts/predict.py',
        'scripts/evaluate.py'
    ]
    
    all_ok = True
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ Directory: {dir_name}/")
        else:
            print(f"✗ Missing directory: {dir_name}/")
            all_ok = False
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✓ File: {file_name}")
        else:
            print(f"✗ Missing file: {file_name}")
            all_ok = False
    
    return all_ok


def test_model_instantiation():
    """Test if model can be instantiated."""
    print("\nTesting model instantiation...")
    
    try:
        # Add parent directory to path
        sys.path.append(str(Path(__file__).parent))
        
        from models.cspan import CSPAN
        
        # Check if config exists
        if not Path('config.yml').exists():
            print("⚠ config.yml not found, skipping model test")
            return False
        
        # Try to create model
        model = CSPAN('config.yml')
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ Model created successfully")
        print(f"  - Total parameters: {total_params:,}")
        
        # Test forward pass with dummy data
        batch_size = 2
        seq_len = 100
        esm_dim = 1280
        
        dummy_input = torch.randn(batch_size, seq_len, esm_dim)
        dummy_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            output = model(dummy_input, dummy_mask)
        
        print(f"✓ Forward pass successful")
        print(f"  - Output shape: {output['probabilities'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {str(e)}")
        return False


def test_data_loading():
    """Test if HuggingFace dataset can be accessed."""
    print("\nTesting data loading...")
    
    try:
        from datasets import load_dataset
        
        # Try to load dataset info
        print("Checking HuggingFace dataset availability...")
        dataset = load_dataset(
            "Synthyra/homodimer_benchmark",
            split="train",
            streaming=True
        )
        
        # Get one example
        example = next(iter(dataset))
        
        print(f"✓ Dataset accessible")
        print(f"  - Example sequence length: {len(example['sequence'])}")
        print(f"  - Example label: {example['label']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {str(e)}")
        print("  Make sure you have internet connection")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("CSPAN Installation Test")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("CUDA Test", test_cuda),
        ("Project Structure Test", test_project_structure),
        ("Model Test", test_model_instantiation),
        ("Data Loading Test", test_data_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed! CSPAN is ready to use.")
        print("\nNext steps:")
        print("1. Run feature extraction: python data/feature_extraction.py")
        print("2. Train model: python scripts/train.py")
        print("3. Or run complete pipeline: python run_pipeline.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Make sure you're in the project root directory")
        print("3. Check internet connection for HuggingFace datasets")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())