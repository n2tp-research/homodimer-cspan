"""
Benchmark different attention implementations for performance comparison.
"""

import torch
import torch.nn as nn
import time
import argparse
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.layers import GlobalFeatureExtractor
from utils.chunked_attention import MemoryEfficientAttention, LinearAttention


def benchmark_attention(
    seq_lengths: List[int],
    batch_size: int = 4,
    d_model: int = 512,
    num_heads: int = 8,
    device: str = 'cuda',
    num_runs: int = 10
) -> Dict[str, Dict[str, List[float]]]:
    """
    Benchmark different attention implementations.
    
    Returns:
        Dictionary with timing and memory results
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    results = {
        'standard': {'time': [], 'memory': []},
        'flash': {'time': [], 'memory': []},
        'chunked': {'time': [], 'memory': []},
        'linear': {'time': [], 'memory': []}
    }
    
    # Check if flash attention is available
    flash_available = False
    try:
        from flash_attn import flash_attn_func
        flash_available = True
        print("Flash Attention is available")
    except ImportError:
        print("Flash Attention not available")
    
    for seq_len in seq_lengths:
        print(f"\nBenchmarking sequence length: {seq_len}")
        
        # Create test input
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        mask = torch.ones(batch_size, seq_len, device=device)
        
        # Create fp16 versions for flash attention
        x_fp16 = x.half()
        mask_fp16 = mask.half()
        
        # Test standard attention
        print("  Testing standard attention...")
        try:
            attn = GlobalFeatureExtractor(
                in_channels=d_model,
                d_model=d_model,
                num_heads=num_heads,
                use_flash_attention=False
            ).to(device)
            
            # Warmup
            for _ in range(3):
                _ = attn(x, mask)
            
            # Time it
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_runs):
                _ = attn(x, mask)
            
            torch.cuda.synchronize()
            elapsed = (time.time() - start_time) / num_runs
            
            # Memory usage
            memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            torch.cuda.reset_peak_memory_stats()
            
            results['standard']['time'].append(elapsed)
            results['standard']['memory'].append(memory)
            print(f"    Time: {elapsed:.4f}s, Memory: {memory:.2f}GB")
            
        except RuntimeError as e:
            print(f"    Failed: {e}")
            results['standard']['time'].append(float('inf'))
            results['standard']['memory'].append(float('inf'))
        
        # Test flash attention
        if flash_available:
            print("  Testing flash attention...")
            try:
                attn = MemoryEfficientAttention(
                    d_model=d_model,
                    num_heads=num_heads,
                    attention_mode='flash'
                ).to(device).half()  # Flash attention requires fp16
                
                # Warmup
                for _ in range(3):
                    _ = attn(x_fp16, mask_fp16)[0]
                
                # Time it
                torch.cuda.synchronize()
                start_time = time.time()
                
                for _ in range(num_runs):
                    _ = attn(x_fp16, mask_fp16)[0]
                
                torch.cuda.synchronize()
                elapsed = (time.time() - start_time) / num_runs
                
                # Memory usage
                memory = torch.cuda.max_memory_allocated() / 1024**3
                torch.cuda.reset_peak_memory_stats()
                
                results['flash']['time'].append(elapsed)
                results['flash']['memory'].append(memory)
                print(f"    Time: {elapsed:.4f}s, Memory: {memory:.2f}GB")
                
            except Exception as e:
                print(f"    Failed: {e}")
                results['flash']['time'].append(float('inf'))
                results['flash']['memory'].append(float('inf'))
        else:
            results['flash']['time'].append(float('inf'))
            results['flash']['memory'].append(float('inf'))
        
        # Test chunked attention
        print("  Testing chunked attention...")
        try:
            attn = MemoryEfficientAttention(
                d_model=d_model,
                num_heads=num_heads,
                attention_mode='chunked'
            ).to(device)
            
            # Warmup
            for _ in range(3):
                _ = attn(x, mask)[0]
            
            # Time it
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_runs):
                _ = attn(x, mask)[0]
            
            torch.cuda.synchronize()
            elapsed = (time.time() - start_time) / num_runs
            
            # Memory usage
            memory = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()
            
            results['chunked']['time'].append(elapsed)
            results['chunked']['memory'].append(memory)
            print(f"    Time: {elapsed:.4f}s, Memory: {memory:.2f}GB")
            
        except Exception as e:
            print(f"    Failed: {e}")
            results['chunked']['time'].append(float('inf'))
            results['chunked']['memory'].append(float('inf'))
        
        # Test linear attention
        print("  Testing linear attention...")
        try:
            attn = LinearAttention(
                d_model=d_model,
                num_heads=num_heads
            ).to(device)
            
            # Warmup
            for _ in range(3):
                _ = attn(x, mask)
            
            # Time it
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_runs):
                _ = attn(x, mask)
            
            torch.cuda.synchronize()
            elapsed = (time.time() - start_time) / num_runs
            
            # Memory usage
            memory = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()
            
            results['linear']['time'].append(elapsed)
            results['linear']['memory'].append(memory)
            print(f"    Time: {elapsed:.4f}s, Memory: {memory:.2f}GB")
            
        except Exception as e:
            print(f"    Failed: {e}")
            results['linear']['time'].append(float('inf'))
            results['linear']['memory'].append(float('inf'))
        
        # Clear cache
        torch.cuda.empty_cache()
    
    return results


def plot_results(results: Dict, seq_lengths: List[int], output_dir: str = "results"):
    """Plot benchmark results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Time comparison
    plt.figure(figsize=(10, 6))
    for method, data in results.items():
        if data['time'][0] != float('inf'):
            plt.plot(seq_lengths, data['time'], marker='o', label=method.capitalize())
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Time per Forward Pass (seconds)')
    plt.title('Attention Mechanism Speed Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(output_dir / 'attention_speed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Memory comparison
    plt.figure(figsize=(10, 6))
    for method, data in results.items():
        if data['memory'][0] != float('inf'):
            plt.plot(seq_lengths, data['memory'], marker='o', label=method.capitalize())
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Peak Memory Usage (GB)')
    plt.title('Attention Mechanism Memory Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(output_dir / 'attention_memory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Speedup comparison
    plt.figure(figsize=(10, 6))
    standard_times = np.array(results['standard']['time'])
    
    for method in ['flash', 'chunked', 'linear']:
        method_times = np.array(results[method]['time'])
        valid_mask = (method_times != float('inf')) & (standard_times != float('inf'))
        
        if valid_mask.any():
            speedup = standard_times[valid_mask] / method_times[valid_mask]
            valid_lengths = np.array(seq_lengths)[valid_mask]
            plt.plot(valid_lengths, speedup, marker='o', label=f'{method.capitalize()} vs Standard')
    
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup Factor')
    plt.title('Attention Speedup Comparison (Higher is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.savefig(output_dir / 'attention_speedup_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark attention mechanisms')
    parser.add_argument('--seq-lengths', nargs='+', type=int,
                       default=[128, 256, 512, 1024, 2048, 4096],
                       help='Sequence lengths to benchmark')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for benchmarking')
    parser.add_argument('--d-model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num-runs', type=int, default=10,
                       help='Number of runs per benchmark')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--output-dir', type=str, default='results/attention_benchmark',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    print("Starting attention mechanism benchmarks...")
    print(f"Configuration:")
    print(f"  Sequence lengths: {args.seq_lengths}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Number of heads: {args.num_heads}")
    print(f"  Device: {args.device}")
    
    # Run benchmarks
    results = benchmark_attention(
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        device=args.device,
        num_runs=args.num_runs
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    
    for seq_len, idx in zip(args.seq_lengths, range(len(args.seq_lengths))):
        print(f"\nSequence Length: {seq_len}")
        print("-" * 40)
        
        for method in results:
            time = results[method]['time'][idx]
            memory = results[method]['memory'][idx]
            
            if time != float('inf'):
                print(f"{method:10s}: {time:.4f}s, {memory:.2f}GB")
            else:
                print(f"{method:10s}: Failed")
    
    # Plot results
    plot_results(results, args.seq_lengths, args.output_dir)
    
    # Recommendations
    print("\n" + "="*60)
    print("Recommendations")
    print("="*60)
    
    if results['flash']['time'][0] != float('inf'):
        print("✓ Flash Attention is available and recommended for sequences up to 2048")
    else:
        print("✗ Flash Attention not available - install with: pip install flash-attn")
    
    print("\nFor different sequence lengths:")
    for seq_len in [512, 1024, 2048, 4096]:
        if seq_len <= 512:
            print(f"  {seq_len:4d} tokens: Standard attention (fast enough)")
        elif seq_len <= 2048 and results['flash']['time'][0] != float('inf'):
            print(f"  {seq_len:4d} tokens: Flash attention (best performance)")
        elif seq_len <= 4096:
            print(f"  {seq_len:4d} tokens: Chunked attention (good memory efficiency)")
        else:
            print(f"  {seq_len:4d} tokens: Linear attention (O(n) complexity)")


if __name__ == '__main__':
    main()