#!/usr/bin/env python3
"""
Triton Kernel Profiling and Analysis
AI Workshop - Version 3

This script profiles Triton kernels and analyzes their performance
characteristics compared to PyTorch native operations.
"""

import torch
import triton
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Import our Triton model
from tiny_llama_v3 import (
    TinyLlamaTriton, ModelConfig,
    TritonRMSNorm, TritonSwiGLU, TritonAttention,
    rmsnorm_kernel, swiglu_kernel, flash_attention_kernel
)


def profile_triton_kernels():
    """Profile individual Triton kernels and compare with PyTorch."""
    print("=== Triton Kernel Performance Analysis ===")

    device = torch.device('cuda')
    results = {}

    # Test configurations
    batch_size = 4
    seq_len = 512
    dim = 2048
    num_runs = 100

    # 1. RMSNorm Kernel Analysis
    print("\n1. RMSNorm Kernel Profiling")

    # Triton RMSNorm
    triton_rmsnorm = TritonRMSNorm(dim).to(device)
    x = torch.randn(batch_size, seq_len, dim, device=device)

    # Warmup
    for _ in range(10):
        _ = triton_rmsnorm(x)
    torch.cuda.synchronize()

    # Benchmark Triton
    start_time = time.time()
    for _ in range(num_runs):
        output_triton = triton_rmsnorm(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / num_runs

    # PyTorch RMSNorm
    class PyTorchRMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return x * self.weight

    pytorch_rmsnorm = PyTorchRMSNorm(dim).to(device)
    pytorch_rmsnorm.weight.data.copy_(triton_rmsnorm.weight.data)

    # Warmup
    for _ in range(10):
        _ = pytorch_rmsnorm(x)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    start_time = time.time()
    for _ in range(num_runs):
        output_pytorch = pytorch_rmsnorm(x)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / num_runs

    rmsnorm_speedup = pytorch_time / triton_time
    results['rmsnorm'] = {
        'triton_time_ms': triton_time * 1000,
        'pytorch_time_ms': pytorch_time * 1000,
        'speedup': rmsnorm_speedup,
        'accuracy_error': torch.abs(output_triton - output_pytorch).max().item()
    }

    print(f"  Triton RMSNorm: {triton_time*1000:.3f} ms")
    print(f"  PyTorch RMSNorm: {pytorch_time*1000:.3f} ms")
    print(f"  Speedup: {rmsnorm_speedup:.2f}x")
    print(f"  Max error: {results['rmsnorm']['accuracy_error']:.2e}")

    # 2. SwiGLU Kernel Analysis
    print("\n2. SwiGLU Kernel Profiling")

    hidden_dim = int(2.67 * dim)
    triton_swiglu = TritonSwiGLU(dim, hidden_dim).to(device)

    # Warmup
    for _ in range(10):
        _ = triton_swiglu(x)
    torch.cuda.synchronize()

    # Benchmark Triton SwiGLU
    start_time = time.time()
    for _ in range(num_runs):
        output_triton = triton_swiglu(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / num_runs

    # PyTorch SwiGLU
    class PyTorchSwiGLU(torch.nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.gate_proj = torch.nn.Linear(dim, hidden_dim, bias=False)
            self.up_proj = torch.nn.Linear(dim, hidden_dim, bias=False)
            self.down_proj = torch.nn.Linear(hidden_dim, dim, bias=False)

        def forward(self, x):
            gate = torch.nn.functional.silu(self.gate_proj(x))
            up = self.up_proj(x)
            return self.down_proj(gate * up)

    pytorch_swiglu = PyTorchSwiGLU(dim, hidden_dim).to(device)

    # Copy weights
    pytorch_swiglu.gate_proj.weight.data.copy_(triton_swiglu.gate_proj.weight.data)
    pytorch_swiglu.up_proj.weight.data.copy_(triton_swiglu.up_proj.weight.data)
    pytorch_swiglu.down_proj.weight.data.copy_(triton_swiglu.down_proj.weight.data)

    # Warmup
    for _ in range(10):
        _ = pytorch_swiglu(x)
    torch.cuda.synchronize()

    # Benchmark PyTorch SwiGLU
    start_time = time.time()
    for _ in range(num_runs):
        output_pytorch = pytorch_swiglu(x)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / num_runs

    swiglu_speedup = pytorch_time / triton_time
    results['swiglu'] = {
        'triton_time_ms': triton_time * 1000,
        'pytorch_time_ms': pytorch_time * 1000,
        'speedup': swiglu_speedup,
        'accuracy_error': torch.abs(output_triton - output_pytorch).max().item()
    }

    print(f"  Triton SwiGLU: {triton_time*1000:.3f} ms")
    print(f"  PyTorch SwiGLU: {pytorch_time*1000:.3f} ms")
    print(f"  Speedup: {swiglu_speedup:.2f}x")
    print(f"  Max error: {results['swiglu']['accuracy_error']:.2e}")

    # 3. Attention Kernel Analysis
    print("\n3. Flash Attention Kernel Profiling")

    n_heads = 32
    triton_attention = TritonAttention(dim, n_heads).to(device)

    # Warmup
    for _ in range(10):
        _ = triton_attention(x)
    torch.cuda.synchronize()

    # Benchmark Triton Attention
    start_time = time.time()
    for _ in range(num_runs // 4):  # Attention is more expensive
        output_triton = triton_attention(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / (num_runs // 4)

    # PyTorch Attention
    class PyTorchAttention(torch.nn.Module):
        def __init__(self, dim, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            self.scale = 1.0 / (self.head_dim ** 0.5)

            self.q_proj = torch.nn.Linear(dim, dim, bias=False)
            self.k_proj = torch.nn.Linear(dim, dim, bias=False)
            self.v_proj = torch.nn.Linear(dim, dim, bias=False)
            self.o_proj = torch.nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            B, T, C = x.shape

            q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

            # Scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Causal mask
            mask = torch.tril(torch.ones(T, T, device=x.device))
            scores = scores.masked_fill(mask == 0, float('-inf'))

            attn = torch.nn.functional.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.o_proj(out)

    pytorch_attention = PyTorchAttention(dim, n_heads).to(device)

    # Copy weights
    pytorch_attention.q_proj.weight.data.copy_(triton_attention.q_proj.weight.data)
    pytorch_attention.k_proj.weight.data.copy_(triton_attention.k_proj.weight.data)
    pytorch_attention.v_proj.weight.data.copy_(triton_attention.v_proj.weight.data)
    pytorch_attention.o_proj.weight.data.copy_(triton_attention.o_proj.weight.data)

    # Warmup
    for _ in range(10):
        _ = pytorch_attention(x)
    torch.cuda.synchronize()

    # Benchmark PyTorch Attention
    start_time = time.time()
    for _ in range(num_runs // 4):
        output_pytorch = pytorch_attention(x)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / (num_runs // 4)

    attention_speedup = pytorch_time / triton_time
    results['attention'] = {
        'triton_time_ms': triton_time * 1000,
        'pytorch_time_ms': pytorch_time * 1000,
        'speedup': attention_speedup,
        'accuracy_error': torch.abs(output_triton - output_pytorch).max().item()
    }

    print(f"  Triton Attention: {triton_time*1000:.3f} ms")
    print(f"  PyTorch Attention: {pytorch_time*1000:.3f} ms")
    print(f"  Speedup: {attention_speedup:.2f}x")
    print(f"  Max error: {results['attention']['accuracy_error']:.2e}")

    # Overall analysis
    print("\n=== Kernel Performance Summary ===")
    total_triton_time = (results['rmsnorm']['triton_time_ms'] +
                        results['swiglu']['triton_time_ms'] +
                        results['attention']['triton_time_ms'])
    total_pytorch_time = (results['rmsnorm']['pytorch_time_ms'] +
                         results['swiglu']['pytorch_time_ms'] +
                         results['attention']['pytorch_time_ms'])

    overall_speedup = total_pytorch_time / total_triton_time
    print(f"Overall speedup: {overall_speedup:.2f}x")
    print(f"Total Triton time: {total_triton_time:.2f} ms")
    print(f"Total PyTorch time: {total_pytorch_time:.2f} ms")

    results['overall'] = {
        'speedup': overall_speedup,
        'total_triton_time_ms': total_triton_time,
        'total_pytorch_time_ms': total_pytorch_time
    }

    return results


def analyze_kernel_efficiency():
    """Analyze kernel efficiency metrics."""
    print("\n=== Kernel Efficiency Analysis ===")

    device = torch.device('cuda')

    # Get GPU properties
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}")
    print(f"Memory: {props.total_memory / 1e9:.1f} GB")
    print(f"SMs: {props.multi_processor_count}")
    print(f"Max threads per SM: {props.max_threads_per_multi_processor}")

    # Analyze memory bandwidth utilization
    batch_size, seq_len, dim = 4, 512, 2048

    # RMSNorm memory analysis
    input_size = batch_size * seq_len * dim * 4  # float32
    weight_size = dim * 4
    output_size = input_size
    total_bytes = input_size + weight_size + output_size

    print(f"\nRMSNorm Memory Analysis:")
    print(f"  Input size: {input_size / 1e6:.2f} MB")
    print(f"  Weight size: {weight_size / 1e3:.2f} KB")
    print(f"  Output size: {output_size / 1e6:.2f} MB")
    print(f"  Total memory: {total_bytes / 1e6:.2f} MB")

    # Theoretical bandwidth (example for MI250X)
    theoretical_bandwidth = 1600e9  # 1.6 TB/s

    # Estimate achieved bandwidth
    x = torch.randn(batch_size, seq_len, dim, device=device)
    triton_rmsnorm = TritonRMSNorm(dim).to(device)

    # Time the operation
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        _ = triton_rmsnorm(x)
    torch.cuda.synchronize()
    avg_time = (time.time() - start_time) / 100

    achieved_bandwidth = total_bytes / avg_time
    bandwidth_efficiency = achieved_bandwidth / theoretical_bandwidth * 100

    print(f"  Execution time: {avg_time*1000:.3f} ms")
    print(f"  Achieved bandwidth: {achieved_bandwidth / 1e9:.1f} GB/s")
    print(f"  Bandwidth efficiency: {bandwidth_efficiency:.1f}%")

    return {
        'bandwidth_efficiency': bandwidth_efficiency,
        'achieved_bandwidth_gbs': achieved_bandwidth / 1e9,
        'theoretical_bandwidth_gbs': theoretical_bandwidth / 1e9
    }


def profile_kernel_launch_overhead():
    """Profile kernel launch overhead and grid configurations."""
    print("\n=== Kernel Launch Overhead Analysis ===")

    device = torch.device('cuda')
    results = {}

    # Test different problem sizes
    problem_sizes = [
        (1, 128, 512),    # Small
        (2, 256, 1024),   # Medium
        (4, 512, 2048),   # Large
        (8, 1024, 4096),  # Extra large
    ]

    for batch_size, seq_len, dim in problem_sizes:
        print(f"\nTesting size: batch={batch_size}, seq={seq_len}, dim={dim}")

        x = torch.randn(batch_size, seq_len, dim, device=device)
        triton_rmsnorm = TritonRMSNorm(dim).to(device)

        # Measure launch overhead
        torch.cuda.synchronize()
        start_time = time.time()

        # Single kernel launch
        output = triton_rmsnorm(x)
        torch.cuda.synchronize()

        single_time = time.time() - start_time

        # Multiple launches
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(10):
            output = triton_rmsnorm(x)
        torch.cuda.synchronize()

        multiple_time = (time.time() - start_time) / 10

        overhead = single_time - multiple_time

        size_key = f"{batch_size}x{seq_len}x{dim}"
        results[size_key] = {
            'single_launch_ms': single_time * 1000,
            'avg_launch_ms': multiple_time * 1000,
            'overhead_ms': overhead * 1000,
            'overhead_percent': (overhead / single_time) * 100 if single_time > 0 else 0
        }

        print(f"  Single launch: {single_time*1000:.3f} ms")
        print(f"  Average launch: {multiple_time*1000:.3f} ms")
        print(f"  Overhead: {overhead*1000:.3f} ms ({results[size_key]['overhead_percent']:.1f}%)")

    return results


def save_profiling_results(kernel_results, efficiency_results, overhead_results):
    """Save all profiling results to JSON."""
    output_dir = Path("profiling_results")
    output_dir.mkdir(exist_ok=True)

    all_results = {
        'kernel_performance': kernel_results,
        'efficiency_analysis': efficiency_results,
        'launch_overhead': overhead_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'gpu_info': {
            'name': torch.cuda.get_device_properties(0).name,
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
            'sm_count': torch.cuda.get_device_properties(0).multi_processor_count
        }
    }

    output_file = output_dir / "triton_profiling_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Create summary report
    summary_file = output_dir / "triton_summary_report.md"
    with open(summary_file, 'w') as f:
        f.write("# Triton Kernel Profiling Summary\n\n")
        f.write(f"**GPU:** {all_results['gpu_info']['name']}\n")
        f.write(f"**Memory:** {all_results['gpu_info']['memory_gb']:.1f} GB\n")
        f.write(f"**Timestamp:** {all_results['timestamp']}\n\n")

        f.write("## Kernel Performance\n\n")
        f.write("| Kernel | Triton (ms) | PyTorch (ms) | Speedup | Error |\n")
        f.write("|--------|-------------|--------------|---------|-------|\n")

        for kernel_name, data in kernel_results.items():
            if kernel_name != 'overall':
                f.write(f"| {kernel_name.title()} | {data['triton_time_ms']:.3f} | "
                       f"{data['pytorch_time_ms']:.3f} | {data['speedup']:.2f}x | "
                       f"{data['accuracy_error']:.2e} |\n")

        f.write(f"\n**Overall Speedup:** {kernel_results['overall']['speedup']:.2f}x\n\n")

        f.write("## Efficiency Analysis\n\n")
        f.write(f"- Bandwidth Efficiency: {efficiency_results['bandwidth_efficiency']:.1f}%\n")
        f.write(f"- Achieved Bandwidth: {efficiency_results['achieved_bandwidth_gbs']:.1f} GB/s\n")
        f.write(f"- Theoretical Bandwidth: {efficiency_results['theoretical_bandwidth_gbs']:.1f} GB/s\n")

    print(f"Summary report saved to: {summary_file}")


if __name__ == "__main__":
    print("Triton Kernel Profiling Suite")
    print("=" * 50)

    # Run all profiling analyses
    kernel_results = profile_triton_kernels()
    efficiency_results = analyze_kernel_efficiency()
    overhead_results = profile_kernel_launch_overhead()

    # Save results
    save_profiling_results(kernel_results, efficiency_results, overhead_results)

    print("\n" + "=" * 50)
    print("Profiling complete! Check profiling_results/ for detailed analysis.")
