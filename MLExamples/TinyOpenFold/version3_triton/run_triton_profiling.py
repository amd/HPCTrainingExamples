#!/usr/bin/env python3
"""
Triton-Specific Profiling Script for TinyOpenFold V3

This script provides comprehensive profiling of Triton kernels including:
- Individual kernel performance analysis
- Memory bandwidth utilization
- Kernel launch overhead
- Comparison with PyTorch baseline operations
"""

import torch
import torch.nn as nn
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# Import V3 model
from tiny_openfold_v3 import (
    TinyOpenFoldV3,
    TinyOpenFoldConfig,
    ProteinDataset,
    TritonLayerNorm,
    TritonMSARowAttention,
    TritonMSAColumnAttention,
    TritonTriangleAttention,
)


def benchmark_kernel(kernel_fn, inputs, num_runs=100, warmup=10):
    """Benchmark a specific kernel or function."""
    # Warmup
    for _ in range(warmup):
        _ = kernel_fn(*inputs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        output = kernel_fn(*inputs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    avg_time = elapsed / num_runs
    
    return avg_time, output


def profile_layernorm(device, dim=128, batch_size=1024):
    """Profile Triton LayerNorm vs PyTorch LayerNorm."""
    print("\n" + "="*70)
    print("LayerNorm Profiling")
    print("="*70)
    
    # Create test data
    x = torch.randn(batch_size, dim, device=device)
    
    # Triton LayerNorm
    triton_norm = TritonLayerNorm(dim).to(device)
    triton_time, triton_output = benchmark_kernel(triton_norm, [x])
    
    # PyTorch LayerNorm
    pytorch_norm = nn.LayerNorm(dim).to(device)
    pytorch_time, pytorch_output = benchmark_kernel(pytorch_norm, [x])
    
    # Check correctness
    rel_error = torch.abs(triton_output - pytorch_output).max() / torch.abs(pytorch_output).max()
    
    print(f"\nLayerNorm Results (dim={dim}, batch={batch_size}):")
    print(f"  Triton:  {triton_time*1000:.3f} ms")
    print(f"  PyTorch: {pytorch_time*1000:.3f} ms")
    print(f"  Speedup: {pytorch_time/triton_time:.2f}x")
    print(f"  Relative Error: {rel_error:.2e}")
    
    return {
        'triton_time_ms': triton_time * 1000,
        'pytorch_time_ms': pytorch_time * 1000,
        'speedup': pytorch_time / triton_time,
        'relative_error': rel_error.item()
    }


def profile_msa_attention(device, config):
    """Profile MSA attention kernels."""
    print("\n" + "="*70)
    print("MSA Attention Profiling")
    print("="*70)
    
    batch_size = 2
    n_seqs = config.n_seqs
    seq_len = config.max_seq_len
    
    # Create test data
    msa = torch.randn(batch_size, n_seqs, seq_len, config.msa_dim, device=device)
    pair = torch.randn(batch_size, seq_len, seq_len, config.pair_dim, device=device)
    
    # Triton MSA Row Attention
    triton_row_attn = TritonMSARowAttention(config).to(device)
    row_time, row_output = benchmark_kernel(triton_row_attn, [msa, pair], num_runs=50)
    
    # Triton MSA Column Attention
    triton_col_attn = TritonMSAColumnAttention(config).to(device)
    col_time, col_output = benchmark_kernel(triton_col_attn, [msa], num_runs=50)
    
    print(f"\nMSA Row Attention (batch={batch_size}, n_seqs={n_seqs}, seq_len={seq_len}):")
    print(f"  Time: {row_time*1000:.3f} ms")
    print(f"  Memory: {msa.element_size() * msa.nelement() / 1e6:.2f} MB input")
    
    print(f"\nMSA Column Attention (batch={batch_size}, n_seqs={n_seqs}, seq_len={seq_len}):")
    print(f"  Time: {col_time*1000:.3f} ms")
    
    return {
        'msa_row_time_ms': row_time * 1000,
        'msa_col_time_ms': col_time * 1000,
        'total_msa_attention_ms': (row_time + col_time) * 1000
    }


def profile_triangle_attention(device, config):
    """Profile Triangle attention kernels."""
    print("\n" + "="*70)
    print("Triangle Attention Profiling")
    print("="*70)
    
    batch_size = 2
    seq_len = config.max_seq_len
    
    # Create test data
    pair = torch.randn(batch_size, seq_len, seq_len, config.pair_dim, device=device)
    
    # Triton Triangle Attention (starting)
    triton_tri_attn_start = TritonTriangleAttention(config, starting=True).to(device)
    start_time, start_output = benchmark_kernel(triton_tri_attn_start, [pair], num_runs=50)
    
    # Triton Triangle Attention (ending)
    triton_tri_attn_end = TritonTriangleAttention(config, starting=False).to(device)
    end_time, end_output = benchmark_kernel(triton_tri_attn_end, [pair], num_runs=50)
    
    print(f"\nTriangle Attention Starting (batch={batch_size}, seq_len={seq_len}):")
    print(f"  Time: {start_time*1000:.3f} ms")
    
    print(f"\nTriangle Attention Ending (batch={batch_size}, seq_len={seq_len}):")
    print(f"  Time: {end_time*1000:.3f} ms")
    
    return {
        'triangle_attn_start_ms': start_time * 1000,
        'triangle_attn_end_ms': end_time * 1000,
        'total_triangle_attention_ms': (start_time + end_time) * 1000
    }


def profile_full_model(device, config, batch_size=4, num_steps=20):
    """Profile the complete V3 model."""
    print("\n" + "="*70)
    print("Full Model Profiling")
    print("="*70)
    
    # Create model and dataset
    model = TinyOpenFoldV3(config).to(device)
    dataset = ProteinDataset(config)
    
    # Warmup
    print(f"\nRunning warmup...")
    for _ in range(5):
        msa_tokens, pair_features, target_distances = dataset.get_batch(batch_size)
        msa_tokens = msa_tokens.to(device)
        pair_features = pair_features.to(device)
        target_distances = target_distances.to(device)
        
        outputs = model(msa_tokens, pair_features, target_distances)
        loss = outputs['loss']
    
    # Profile forward pass
    print(f"Profiling forward pass...")
    forward_times = []
    memory_usage = []
    
    for _ in range(num_steps):
        msa_tokens, pair_features, target_distances = dataset.get_batch(batch_size)
        msa_tokens = msa_tokens.to(device)
        pair_features = pair_features.to(device)
        target_distances = target_distances.to(device)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        outputs = model(msa_tokens, pair_features, target_distances)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        forward_times.append(time.time() - start)
        
        if torch.cuda.is_available():
            memory_usage.append(torch.cuda.memory_allocated() / 1e6)
    
    avg_forward = np.mean(forward_times)
    avg_memory = np.mean(memory_usage)
    
    print(f"\nFull Model Results (batch={batch_size}, {num_steps} iterations):")
    print(f"  Avg Forward Time: {avg_forward*1000:.3f} ms")
    print(f"  Throughput: {batch_size / avg_forward:.1f} samples/sec")
    if memory_usage:
        print(f"  Avg Memory: {avg_memory:.1f} MB")
        print(f"  Peak Memory: {max(memory_usage):.1f} MB")
    
    return {
        'avg_forward_time_ms': avg_forward * 1000,
        'throughput_samples_per_sec': batch_size / avg_forward,
        'avg_memory_mb': avg_memory if memory_usage else 0,
        'peak_memory_mb': max(memory_usage) if memory_usage else 0
    }


def main():
    parser = argparse.ArgumentParser(description='Triton Profiling for TinyOpenFold V3')
    parser.add_argument('--output-dir', type=str, default='profiling_results',
                        help='Directory to save profiling results')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for profiling')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuration
    config = TinyOpenFoldConfig()
    
    # Run profiling
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'config': config.to_dict()
    }
    
    # Profile individual kernels
    results['layernorm'] = profile_layernorm(device)
    results['msa_attention'] = profile_msa_attention(device, config)
    results['triangle_attention'] = profile_triangle_attention(device, config)
    results['full_model'] = profile_full_model(device, config, args.batch_size)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"triton_profiling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*70)
    print(f"Profiling complete! Results saved to: {output_file}")
    print("="*70)
    
    # Summary
    print(f"\nSummary:")
    print(f"  LayerNorm Speedup: {results['layernorm']['speedup']:.2f}x")
    print(f"  Full Model Throughput: {results['full_model']['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"  Peak Memory: {results['full_model']['peak_memory_mb']:.1f} MB")


if __name__ == "__main__":
    main()

