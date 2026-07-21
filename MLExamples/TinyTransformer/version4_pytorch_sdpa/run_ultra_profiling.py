#!/usr/bin/env python3
"""
Ultra-Fused Performance Profiling and Analysis
AI Workshop - Version 4

This script provides comprehensive profiling and analysis for the ultra-fused
Triton implementation, measuring the ultimate performance achievements.
"""

import torch
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
import psutil
import gc

# Import our ultra-fused model
from tiny_llama_v4 import TinyLlamaUltraFused, ModelConfig


class UltraPerformanceProfiler:
    """Comprehensive profiler for ultra-fused model performance."""

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device)
        self.results = {}
        self.baseline_cache = {}

    def profile_end_to_end_performance(self) -> Dict[str, Any]:
        """Profile complete end-to-end model performance."""
        print("=== End-to-End Performance Profiling ===")

        config = ModelConfig(
            vocab_size=32000,
            dim=2048,
            n_layers=16,
            n_heads=32,
            max_seq_len=2048
        )

        model = TinyLlamaUltraFused(config).to(self.device)

        # Test configurations covering various use cases
        test_configs = [
            ("micro", 1, 32),      # Micro batch
            ("small", 1, 128),     # Single sequence
            ("medium", 2, 256),    # Small batch
            ("large", 4, 512),     # Standard batch
            ("xlarge", 8, 512),    # Large batch
            ("long", 2, 1024),     # Long sequence
            ("ultra_long", 1, 2048), # Maximum sequence
        ]

        results = {}

        for name, batch_size, seq_len in test_configs:
            print(f"\nProfiling {name}: B={batch_size}, S={seq_len}")

            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=self.device)

            # Profile ultra-fused mode
            ultra_metrics = self._profile_single_config(model, input_ids, ultra_mode=True)

            # Profile standard mode for comparison
            standard_metrics = self._profile_single_config(model, input_ids, ultra_mode=False)

            # Calculate improvements
            speedup = standard_metrics['avg_time'] / ultra_metrics['avg_time']
            memory_reduction = (standard_metrics['peak_memory'] - ultra_metrics['peak_memory']) / standard_metrics['peak_memory']

            results[name] = {
                'config': (batch_size, seq_len),
                'ultra': ultra_metrics,
                'standard': standard_metrics,
                'speedup': speedup,
                'memory_reduction': memory_reduction,
                'throughput_improvement': (ultra_metrics['throughput'] - standard_metrics['throughput']) / standard_metrics['throughput']
            }

            print(f"  Ultra-fused: {ultra_metrics['avg_time']*1000:.2f} ms, {ultra_metrics['peak_memory']/1e9:.2f} GB")
            print(f"  Standard: {standard_metrics['avg_time']*1000:.2f} ms, {standard_metrics['peak_memory']/1e9:.2f} GB")
            print(f"  Improvement: {speedup:.2f}x speedup, {memory_reduction*100:.1f}% memory reduction")

        self.results['end_to_end'] = results
        return results

    def _profile_single_config(self, model, input_ids, ultra_mode: bool, num_runs: int = 50) -> Dict[str, float]:
        """Profile a single configuration."""
        model.enable_ultra_mode(ultra_mode)
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids)

        torch.cuda.synchronize()

        # Clear memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Benchmark
        times = []
        start_time = time.time()

        with torch.no_grad():
            for i in range(num_runs):
                iter_start = time.time()
                outputs = model(input_ids)
                torch.cuda.synchronize()
                iter_time = time.time() - iter_start
                times.append(iter_time)

        total_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated()

        # Calculate statistics
        times = np.array(times[5:])  # Skip first 5 for stability
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        p95_time = np.percentile(times, 95)

        batch_size, seq_len = input_ids.shape
        throughput = batch_size * seq_len / avg_time

        return {
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'p95_time': p95_time,
            'peak_memory': peak_memory,
            'throughput': throughput,
            'total_time': total_time
        }

    def profile_scaling_behavior(self) -> Dict[str, Any]:
        """Profile how performance scales with model size and sequence length."""
        print("\n=== Scaling Behavior Analysis ===")

        # Sequence length scaling
        seq_scaling = self._profile_sequence_scaling()

        # Batch size scaling
        batch_scaling = self._profile_batch_scaling()

        # Model size scaling (different layer counts)
        model_scaling = self._profile_model_scaling()

        results = {
            'sequence_scaling': seq_scaling,
            'batch_scaling': batch_scaling,
            'model_scaling': model_scaling
        }

        self.results['scaling'] = results
        return results

    def _profile_sequence_scaling(self) -> Dict[str, Any]:
        """Profile performance vs sequence length."""
        print("\nSequence Length Scaling:")

        config = ModelConfig(dim=1024, n_layers=8, n_heads=16)  # Smaller for testing
        model = TinyLlamaUltraFused(config).to(self.device)

        seq_lengths = [64, 128, 256, 512, 1024, 2048]
        batch_size = 2

        results = []

        for seq_len in seq_lengths:
            if seq_len > config.max_seq_len:
                continue

            print(f"  Testing seq_len={seq_len}")

            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=self.device)

            ultra_metrics = self._profile_single_config(model, input_ids, ultra_mode=True, num_runs=20)
            standard_metrics = self._profile_single_config(model, input_ids, ultra_mode=False, num_runs=20)

            results.append({
                'seq_len': seq_len,
                'ultra_time': ultra_metrics['avg_time'],
                'standard_time': standard_metrics['avg_time'],
                'speedup': standard_metrics['avg_time'] / ultra_metrics['avg_time'],
                'ultra_memory': ultra_metrics['peak_memory'],
                'standard_memory': standard_metrics['peak_memory'],
                'throughput': ultra_metrics['throughput']
            })

        return results

    def _profile_batch_scaling(self) -> Dict[str, Any]:
        """Profile performance vs batch size."""
        print("\nBatch Size Scaling:")

        config = ModelConfig(dim=1024, n_layers=8, n_heads=16)
        model = TinyLlamaUltraFused(config).to(self.device)

        batch_sizes = [1, 2, 4, 8, 16]
        seq_len = 256

        results = []

        for batch_size in batch_sizes:
            print(f"  Testing batch_size={batch_size}")

            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=self.device)

            try:
                ultra_metrics = self._profile_single_config(model, input_ids, ultra_mode=True, num_runs=20)
                standard_metrics = self._profile_single_config(model, input_ids, ultra_mode=False, num_runs=20)

                results.append({
                    'batch_size': batch_size,
                    'ultra_time': ultra_metrics['avg_time'],
                    'standard_time': standard_metrics['avg_time'],
                    'speedup': standard_metrics['avg_time'] / ultra_metrics['avg_time'],
                    'ultra_memory': ultra_metrics['peak_memory'],
                    'standard_memory': standard_metrics['peak_memory'],
                    'throughput': ultra_metrics['throughput']
                })

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"    OOM at batch_size={batch_size}")
                    break
                else:
                    raise

        return results

    def _profile_model_scaling(self) -> Dict[str, Any]:
        """Profile performance vs model size."""
        print("\nModel Size Scaling:")

        model_configs = [
            ("tiny", ModelConfig(dim=512, n_layers=4, n_heads=8)),
            ("small", ModelConfig(dim=1024, n_layers=8, n_heads=16)),
            ("medium", ModelConfig(dim=1536, n_layers=12, n_heads=24)),
            ("large", ModelConfig(dim=2048, n_layers=16, n_heads=32)),
        ]

        batch_size, seq_len = 2, 256
        results = []

        for name, config in model_configs:
            print(f"  Testing {name} model: dim={config.dim}, layers={config.n_layers}")

            try:
                model = TinyLlamaUltraFused(config).to(self.device)
                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=self.device)

                ultra_metrics = self._profile_single_config(model, input_ids, ultra_mode=True, num_runs=15)
                standard_metrics = self._profile_single_config(model, input_ids, ultra_mode=False, num_runs=15)

                param_count = sum(p.numel() for p in model.parameters())

                results.append({
                    'model_name': name,
                    'dim': config.dim,
                    'n_layers': config.n_layers,
                    'param_count': param_count,
                    'ultra_time': ultra_metrics['avg_time'],
                    'standard_time': standard_metrics['avg_time'],
                    'speedup': standard_metrics['avg_time'] / ultra_metrics['avg_time'],
                    'ultra_memory': ultra_metrics['peak_memory'],
                    'standard_memory': standard_metrics['peak_memory'],
                    'throughput': ultra_metrics['throughput']
                })

                del model  # Free memory
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"    OOM for {name} model")
                    continue
                else:
                    raise

        return results

    def profile_kernel_efficiency(self) -> Dict[str, Any]:
        """Profile individual kernel efficiency and characteristics."""
        print("\n=== Kernel Efficiency Analysis ===")

        # This would require more detailed kernel-level profiling
        # For now, we'll estimate based on overall performance

        config = ModelConfig(dim=1024, n_layers=4, n_heads=16)
        model = TinyLlamaUltraFused(config).to(self.device)

        batch_size, seq_len = 4, 256
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=self.device)

        # Estimate kernel characteristics
        model.enable_ultra_mode(True)

        # Profile with PyTorch profiler for kernel-level details
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                for _ in range(5):
                    _ = model(input_ids)

        # Analyze profiler results
        key_averages = prof.key_averages()
        kernel_stats = []

        for event in key_averages:
            if event.device_type == torch.profiler.DeviceType.CUDA:
                kernel_stats.append({
                    'name': event.key,
                    'self_cuda_time': event.self_cuda_time,
                    'cuda_time': event.cuda_time,
                    'count': event.count,
                    'self_cpu_time': event.self_cpu_time
                })

        # Sort by CUDA time
        kernel_stats.sort(key=lambda x: x['cuda_time'], reverse=True)

        results = {
            'total_kernels': len(kernel_stats),
            'top_kernels': kernel_stats[:10],  # Top 10 most expensive
            'total_cuda_time': sum(k['cuda_time'] for k in kernel_stats),
            'kernel_efficiency_score': self._calculate_kernel_efficiency_score(kernel_stats)
        }

        self.results['kernel_efficiency'] = results
        return results

    def _calculate_kernel_efficiency_score(self, kernel_stats: List[Dict]) -> float:
        """Calculate a kernel efficiency score based on launch overhead and utilization."""
        if not kernel_stats:
            return 0.0

        total_time = sum(k['cuda_time'] for k in kernel_stats)
        total_launches = sum(k['count'] for k in kernel_stats)

        # Penalize many small kernels, reward fewer large kernels
        avg_kernel_time = total_time / total_launches if total_launches > 0 else 0
        efficiency_score = min(avg_kernel_time / 1000, 1.0)  # Normalize to 0-1

        return efficiency_score

    def generate_performance_report(self) -> None:
        """Generate comprehensive performance report."""
        print("\n=== Generating Performance Report ===")

        output_dir = Path("ultra_profiling_results")
        output_dir.mkdir(exist_ok=True)

        # Save raw results
        with open(output_dir / "ultra_profiling_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Generate markdown report
        self._generate_markdown_report(output_dir)

        # Generate performance plots
        self._generate_performance_plots(output_dir)

        print(f"Performance report saved to: {output_dir}")

    def _generate_markdown_report(self, output_dir: Path) -> None:
        """Generate markdown performance report."""
        report_file = output_dir / "ultra_performance_report.md"

        with open(report_file, 'w') as f:
            f.write("# Ultra-Fused Model Performance Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Device**: {torch.cuda.get_device_name()}\n\n")

            # End-to-end performance
            if 'end_to_end' in self.results:
                f.write("## End-to-End Performance\n\n")
                f.write("| Configuration | Speedup | Memory Reduction | Throughput (tokens/s) |\n")
                f.write("|---------------|---------|------------------|----------------------|\n")

                for name, data in self.results['end_to_end'].items():
                    speedup = data['speedup']
                    mem_reduction = data['memory_reduction'] * 100
                    throughput = data['ultra']['throughput']

                    f.write(f"| {name} | {speedup:.2f}x | {mem_reduction:.1f}% | {throughput:.0f} |\n")

            # Scaling behavior
            if 'scaling' in self.results:
                f.write("\n## Scaling Behavior\n\n")

                if 'sequence_scaling' in self.results['scaling']:
                    f.write("### Sequence Length Scaling\n\n")
                    f.write("| Seq Length | Ultra Time (ms) | Standard Time (ms) | Speedup |\n")
                    f.write("|------------|-----------------|-------------------|----------|\n")

                    for data in self.results['scaling']['sequence_scaling']:
                        ultra_time = data['ultra_time'] * 1000
                        standard_time = data['standard_time'] * 1000
                        speedup = data['speedup']

                        f.write(f"| {data['seq_len']} | {ultra_time:.2f} | {standard_time:.2f} | {speedup:.2f}x |\n")

            # Kernel efficiency
            if 'kernel_efficiency' in self.results:
                f.write("\n## Kernel Efficiency\n\n")
                ke = self.results['kernel_efficiency']
                f.write(f"- **Total Kernels**: {ke['total_kernels']}\n")
                f.write(f"- **Efficiency Score**: {ke['kernel_efficiency_score']:.3f}\n")
                f.write(f"- **Total CUDA Time**: {ke['total_cuda_time']/1000:.2f} ms\n\n")

        print(f"Markdown report saved to: {report_file}")

    def _generate_performance_plots(self, output_dir: Path) -> None:
        """Generate performance visualization plots."""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            pass

        # Speedup comparison plot
        if 'end_to_end' in self.results:
            self._plot_speedup_comparison(output_dir)

        # Scaling plots
        if 'scaling' in self.results:
            self._plot_scaling_behavior(output_dir)

        print("Performance plots generated")

    def _plot_speedup_comparison(self, output_dir: Path) -> None:
        """Plot speedup comparison across configurations."""
        data = self.results['end_to_end']

        configs = list(data.keys())
        speedups = [data[config]['speedup'] for config in configs]
        memory_reductions = [data[config]['memory_reduction'] * 100 for config in configs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Speedup plot
        bars1 = ax1.bar(configs, speedups, color='skyblue', alpha=0.8)
        ax1.set_ylabel('Speedup (x)')
        ax1.set_title('Ultra-Fused vs Standard Performance')
        ax1.set_ylim(0, max(speedups) * 1.1)

        # Add value labels on bars
        for bar, speedup in zip(bars1, speedups):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{speedup:.2f}x', ha='center', va='bottom')

        # Memory reduction plot
        bars2 = ax2.bar(configs, memory_reductions, color='lightcoral', alpha=0.8)
        ax2.set_ylabel('Memory Reduction (%)')
        ax2.set_title('Memory Usage Reduction')
        ax2.set_ylim(0, max(memory_reductions) * 1.1)

        # Add value labels
        for bar, reduction in zip(bars2, memory_reductions):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{reduction:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / "speedup_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_scaling_behavior(self, output_dir: Path) -> None:
        """Plot scaling behavior analysis."""
        scaling_data = self.results['scaling']

        if 'sequence_scaling' in scaling_data:
            seq_data = scaling_data['sequence_scaling']
            seq_lengths = [d['seq_len'] for d in seq_data]
            ultra_times = [d['ultra_time'] * 1000 for d in seq_data]
            standard_times = [d['standard_time'] * 1000 for d in seq_data]

            plt.figure(figsize=(10, 6))
            plt.plot(seq_lengths, ultra_times, 'o-', label='Ultra-Fused', linewidth=2)
            plt.plot(seq_lengths, standard_times, 's-', label='Standard', linewidth=2)
            plt.xlabel('Sequence Length')
            plt.ylabel('Execution Time (ms)')
            plt.title('Performance vs Sequence Length')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / "sequence_scaling.png", dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """Run comprehensive ultra-fused performance profiling."""
    print("Ultra-Fused Performance Profiling Suite")
    print("=" * 50)

    profiler = UltraPerformanceProfiler()

    # Run all profiling analyses
    try:
        profiler.profile_end_to_end_performance()
        profiler.profile_scaling_behavior()
        profiler.profile_kernel_efficiency()

        # Generate comprehensive report
        profiler.generate_performance_report()

        print("\n" + "=" * 50)
        print("Ultra-fused profiling complete!")
        print("Check ultra_profiling_results/ for detailed analysis.")

    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
