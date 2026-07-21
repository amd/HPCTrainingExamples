#!/usr/bin/env python3
"""
DeepSpeed FLOPS Profiler Integration for Tiny LLaMA V1

This script provides comprehensive FLOPS analysis using DeepSpeed's FLOPS profiler
to measure computational efficiency and identify optimization opportunities.

Features:
- Detailed FLOPS breakdown by operation type
- Model FLOPS Utilization (MFU) calculation
- Computational intensity analysis
- Memory bandwidth requirements
- Arithmetic intensity metrics
- Roofline model preparation data

Usage:
    # Run FLOPS profiling with default settings
    python run_deepspeed_flops.py

    # Custom configuration
    python run_deepspeed_flops.py --batch-size 16 --seq-len 256

    # Analyze existing results
    python run_deepspeed_flops.py --analyze-results flops_profile.json

    # Generate roofline analysis data
    python run_deepspeed_flops.py --generate-roofline --output-dir ./roofline_data
"""

import torch
import torch.nn as nn
import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time

# Import the model from tiny_llama_v1
from tiny_llama_v1 import TinyLlama, TinyLlamaConfig, SimpleTextDataset, setup_deterministic_environment

# Optional DeepSpeed import
try:
    from deepspeed.profiling.flops_profiler import FlopsProfiler
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False


class FLOPSAnalyzer:
    """Comprehensive FLOPS analysis and computational efficiency measurement."""

    def __init__(self, output_dir: str = "./flops_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_results = {}

    def profile_model_flops(
        self,
        config: TinyLlamaConfig,
        batch_size: int = 8,
        num_steps: int = 10,
        detailed_analysis: bool = True
    ) -> Dict[str, Any]:
        """Profile model FLOPS using DeepSpeed profiler."""

        if not DEEPSPEED_AVAILABLE:
            return {'error': 'DeepSpeed not available for FLOPS profiling'}

        print(f"Starting FLOPS Analysis")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Batch size: {batch_size}")
        print(f"   Sequence length: {config.max_seq_len}")
        print(f"   Analysis steps: {num_steps}")

        # Setup environment
        setup_deterministic_environment()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model and dataset
        model = TinyLlama(config).to(device)
        dataset = SimpleTextDataset(config.max_seq_len, config.vocab_size)

        # Initialize FLOPS profiler
        prof = FlopsProfiler(model)

        # Model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nModel Information:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size (FP32): {total_params * 4 / 1e6:.1f} MB")

        # Run profiling
        model.train()
        prof.start_profile()

        total_flops = 0
        total_time = 0
        step_results = []

        for step in range(num_steps):
            # Get batch
            input_ids, labels = dataset.get_batch(batch_size)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Time the forward pass
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Forward pass
            outputs = model(input_ids, labels)
            loss = outputs['loss']

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_time = time.time() - start_time

            # Backward pass (for training scenario)
            loss.backward()

            # Get step FLOPS
            if hasattr(prof, 'get_total_flops'):
                step_flops = prof.get_total_flops()
            else:
                # Fallback estimation
                step_flops = self._estimate_transformer_flops(config, batch_size)

            total_flops += step_flops
            total_time += step_time

            step_results.append({
                'step': step,
                'loss': loss.item(),
                'flops': step_flops,
                'time': step_time,
                'flops_per_sec': step_flops / step_time if step_time > 0 else 0
            })

            if step % 2 == 0:
                print(f"   Step {step}: Loss {loss.item():.4f}, "
                      f"FLOPS {step_flops:.2e}, Time {step_time*1000:.1f}ms")

            # Clear gradients for next step
            model.zero_grad()

        # Stop profiling and get results
        prof.stop_profile()

        # Get detailed profile information
        try:
            flops_summary = prof.get_total_flops()
            params_summary = prof.get_total_params()

            if detailed_analysis and hasattr(prof, 'print_model_profile'):
                # Capture detailed profile output
                import io
                import contextlib

                profile_output = io.StringIO()
                with contextlib.redirect_stdout(profile_output):
                    prof.print_model_profile(profile_step=1, module_depth=-1, top_modules=50)

                detailed_profile = profile_output.getvalue()
            else:
                detailed_profile = "Detailed profile not available"

        except Exception as e:
            print(f"   Warning: Could not get detailed FLOPS data: {e}")
            flops_summary = total_flops / num_steps if num_steps > 0 else 0
            params_summary = total_params
            detailed_profile = f"Profile generation failed: {e}"

        # Calculate efficiency metrics
        avg_time_per_step = total_time / num_steps if num_steps > 0 else 0
        avg_flops_per_step = total_flops / num_steps if num_steps > 0 else 0
        throughput = batch_size / avg_time_per_step if avg_time_per_step > 0 else 0

        # Calculate Model FLOPS Utilization (MFU)
        mfu_metrics = self._calculate_mfu(
            model_flops=avg_flops_per_step,
            time_per_step=avg_time_per_step,
            device_peak_flops=self._get_device_peak_flops()
        )

        results = {
            'model_info': {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'config': config.to_dict()
            },
            'profiling_config': {
                'batch_size': batch_size,
                'sequence_length': config.max_seq_len,
                'num_steps': num_steps,
                'device': str(device)
            },
            'flops_analysis': {
                'total_flops': flops_summary,
                'avg_flops_per_step': avg_flops_per_step,
                'flops_per_parameter': avg_flops_per_step / max(1, total_params),
                'detailed_profile': detailed_profile
            },
            'performance_metrics': {
                'avg_time_per_step': avg_time_per_step,
                'throughput_samples_per_sec': throughput,
                'avg_loss': np.mean([r['loss'] for r in step_results]),
                'flops_per_sec': avg_flops_per_step / avg_time_per_step if avg_time_per_step > 0 else 0
            },
            'efficiency_metrics': mfu_metrics,
            'step_by_step_results': step_results,
            'timestamp': datetime.now().isoformat()
        }

        # Save results
        results_path = self.output_dir / "flops_profile.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nFLOPS Analysis Summary:")
        print(f"   Total FLOPS per step: {avg_flops_per_step:.2e}")
        print(f"   FLOPS per parameter: {results['flops_analysis']['flops_per_parameter']:.2f}")
        print(f"   Throughput: {throughput:.1f} samples/sec")
        print(f"   Model FLOPS Utilization: {mfu_metrics['mfu_percent']:.1f}%")
        print(f"   Results saved to: {results_path}")

        return results

    def _estimate_transformer_flops(self, config: TinyLlamaConfig, batch_size: int) -> float:
        """Estimate FLOPS for transformer model (fallback if DeepSpeed fails)."""
        seq_len = config.max_seq_len
        hidden_dim = config.hidden_dim
        n_layers = config.n_layers
        n_heads = config.n_heads
        intermediate_dim = config.intermediate_dim
        vocab_size = config.vocab_size

        # Embedding FLOPS (negligible for most cases)
        embed_flops = 0

        # Per-layer FLOPS
        layer_flops = 0

        # Attention FLOPS
        # QKV projection: 3 * batch_size * seq_len * hidden_dim * hidden_dim
        qkv_flops = 3 * batch_size * seq_len * hidden_dim * hidden_dim

        # Attention scores: batch_size * n_heads * seq_len * seq_len * (hidden_dim / n_heads)
        attention_scores_flops = batch_size * n_heads * seq_len * seq_len * (hidden_dim // n_heads)

        # Attention output: batch_size * n_heads * seq_len * (hidden_dim / n_heads) * seq_len
        attention_output_flops = batch_size * n_heads * seq_len * (hidden_dim // n_heads) * seq_len

        # Output projection: batch_size * seq_len * hidden_dim * hidden_dim
        output_proj_flops = batch_size * seq_len * hidden_dim * hidden_dim

        attention_total = qkv_flops + attention_scores_flops + attention_output_flops + output_proj_flops

        # Feed-forward FLOPS
        # Gate projection: batch_size * seq_len * hidden_dim * intermediate_dim
        gate_flops = batch_size * seq_len * hidden_dim * intermediate_dim

        # Up projection: batch_size * seq_len * hidden_dim * intermediate_dim
        up_flops = batch_size * seq_len * hidden_dim * intermediate_dim

        # Down projection: batch_size * seq_len * intermediate_dim * hidden_dim
        down_flops = batch_size * seq_len * intermediate_dim * hidden_dim

        ffn_total = gate_flops + up_flops + down_flops

        # Layer normalization (minimal)
        norm_flops = 2 * batch_size * seq_len * hidden_dim  # RMSNorm

        layer_flops = attention_total + ffn_total + norm_flops

        # Total model FLOPS
        total_flops = embed_flops + (n_layers * layer_flops)

        # Output projection FLOPS
        output_flops = batch_size * seq_len * hidden_dim * vocab_size
        total_flops += output_flops

        return total_flops

    def _get_device_peak_flops(self) -> float:
        """Get peak FLOPS for the current device."""
        if not torch.cuda.is_available():
            return 1e12  # Rough CPU estimate

        device_name = torch.cuda.get_device_name(0).lower()

        # AMD GPU peak FLOPS (FP32)
        amd_peak_flops = {
            'mi100': 11.5e12,      # 11.5 TFLOPS
            'mi200': 47.9e12,      # 47.9 TFLOPS
            'mi250': 47.9e12,      # 47.9 TFLOPS
            'mi300': 61.3e12,      # 61.3 TFLOPS
            'rx 7900': 61.4e12,    # 61.4 TFLOPS
            'rx 6900': 23.0e12,    # 23.0 TFLOPS
        }

        # NVIDIA GPU peak FLOPS (FP32)
        nvidia_peak_flops = {
            'h100': 67.0e12,       # 67 TFLOPS
            'a100': 19.5e12,       # 19.5 TFLOPS
            'v100': 15.7e12,       # 15.7 TFLOPS
            'rtx 4090': 83.0e12,   # 83 TFLOPS
            'rtx 3090': 35.6e12,   # 35.6 TFLOPS
        }

        # Check AMD GPUs
        for gpu_name, flops in amd_peak_flops.items():
            if gpu_name in device_name:
                return flops

        # Check NVIDIA GPUs
        for gpu_name, flops in nvidia_peak_flops.items():
            if gpu_name in device_name:
                return flops

        # Default fallback
        return 20e12  # 20 TFLOPS as reasonable default

    def _calculate_mfu(self, model_flops: float, time_per_step: float, device_peak_flops: float) -> Dict[str, float]:
        """Calculate Model FLOPS Utilization and related efficiency metrics."""
        if time_per_step <= 0 or device_peak_flops <= 0:
            return {
                'mfu_percent': 0.0,
                'achieved_flops_per_sec': 0.0,
                'device_peak_flops': device_peak_flops,
                'efficiency_ratio': 0.0
            }

        achieved_flops_per_sec = model_flops / time_per_step
        mfu_percent = (achieved_flops_per_sec / device_peak_flops) * 100
        efficiency_ratio = achieved_flops_per_sec / device_peak_flops

        return {
            'mfu_percent': mfu_percent,
            'achieved_flops_per_sec': achieved_flops_per_sec,
            'device_peak_flops': device_peak_flops,
            'efficiency_ratio': efficiency_ratio
        }

    def analyze_computational_intensity(self, flops_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze computational intensity and memory bandwidth requirements."""
        print(f"\nAnalyzing computational intensity...")

        if not torch.cuda.is_available():
            return {'error': 'CUDA not available for memory bandwidth analysis'}

        # Get model info
        model_info = flops_data.get('model_info', {})
        perf_metrics = flops_data.get('performance_metrics', {})
        total_params = model_info.get('total_params', 0)

        # Estimate memory bandwidth requirements
        param_size_bytes = total_params * 4  # FP32
        activation_size_estimate = flops_data['profiling_config']['batch_size'] * \
                                 flops_data['profiling_config']['sequence_length'] * \
                                 model_info['config']['hidden_dim'] * 4  # FP32

        # Memory transfers per step (rough estimate)
        # Parameters read once, activations multiple times
        memory_bytes_per_step = param_size_bytes + (activation_size_estimate * 3)  # Forward + 2x backward

        avg_time = perf_metrics.get('avg_time_per_step', 1.0)
        memory_bandwidth_used = memory_bytes_per_step / avg_time if avg_time > 0 else 0

        # Arithmetic intensity (FLOPS per byte)
        avg_flops = flops_data['flops_analysis']['avg_flops_per_step']
        arithmetic_intensity = avg_flops / memory_bytes_per_step if memory_bytes_per_step > 0 else 0

        # Get device memory bandwidth (rough estimates)
        device_memory_bandwidth = self._get_device_memory_bandwidth()

        intensity_analysis = {
            'arithmetic_intensity_flops_per_byte': arithmetic_intensity,
            'memory_bandwidth_used_gb_per_sec': memory_bandwidth_used / 1e9,
            'memory_bandwidth_utilization_percent': (memory_bandwidth_used / device_memory_bandwidth) * 100,
            'device_memory_bandwidth_gb_per_sec': device_memory_bandwidth / 1e9,
            'memory_bound_vs_compute_bound': 'memory_bound' if arithmetic_intensity < 10 else 'compute_bound',
            'roofline_metrics': {
                'peak_flops': flops_data['efficiency_metrics']['device_peak_flops'],
                'peak_memory_bandwidth': device_memory_bandwidth,
                'achieved_flops': perf_metrics.get('flops_per_sec', 0),
                'achieved_bandwidth': memory_bandwidth_used
            }
        }

        # Save intensity analysis
        intensity_path = self.output_dir / "computational_intensity.json"
        with open(intensity_path, 'w') as f:
            json.dump(intensity_analysis, f, indent=2)

        print(f"   Arithmetic Intensity: {arithmetic_intensity:.2f} FLOPS/byte")
        print(f"   Memory Bandwidth Used: {memory_bandwidth_used/1e9:.1f} GB/s")
        print(f"   Memory vs Compute: {intensity_analysis['memory_bound_vs_compute_bound']}")
        print(f"   Results saved to: {intensity_path}")

        return intensity_analysis

    def _get_device_memory_bandwidth(self) -> float:
        """Get peak memory bandwidth for the current device."""
        if not torch.cuda.is_available():
            return 100e9  # 100 GB/s rough CPU estimate

        device_name = torch.cuda.get_device_name(0).lower()

        # AMD GPU memory bandwidth
        amd_bandwidth = {
            'mi100': 1228e9,      # 1228 GB/s (HBM2)
            'mi200': 1638e9,      # 1638 GB/s (HBM2e)
            'mi250': 1638e9,      # 1638 GB/s (HBM2e)
            'mi300': 5200e9,      # 5200 GB/s (HBM3)
            'rx 7900': 960e9,     # 960 GB/s (GDDR6)
            'rx 6900': 512e9,     # 512 GB/s (GDDR6)
        }

        # NVIDIA GPU memory bandwidth
        nvidia_bandwidth = {
            'h100': 3350e9,       # 3350 GB/s (HBM3)
            'a100': 2039e9,       # 2039 GB/s (HBM2e)
            'v100': 1555e9,       # 1555 GB/s (HBM2)
            'rtx 4090': 1008e9,   # 1008 GB/s (GDDR6X)
            'rtx 3090': 936e9,    # 936 GB/s (GDDR6X)
        }

        # Check AMD GPUs
        for gpu_name, bandwidth in amd_bandwidth.items():
            if gpu_name in device_name:
                return bandwidth

        # Check NVIDIA GPUs
        for gpu_name, bandwidth in nvidia_bandwidth.items():
            if gpu_name in device_name:
                return bandwidth

        # Default fallback
        return 1000e9  # 1000 GB/s as reasonable default

    def generate_roofline_data(self, output_dir: str = None) -> str:
        """Generate data for roofline model analysis."""
        if output_dir is None:
            output_dir = self.output_dir

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load existing analysis results
        flops_file = self.output_dir / "flops_profile.json"
        intensity_file = self.output_dir / "computational_intensity.json"

        if not flops_file.exists():
            return "Error: Run FLOPS profiling first"

        with open(flops_file, 'r') as f:
            flops_data = json.load(f)

        intensity_data = {}
        if intensity_file.exists():
            with open(intensity_file, 'r') as f:
                intensity_data = json.load(f)

        # Prepare roofline data
        roofline_data = {
            'model_name': 'Tiny LLaMA V1 Baseline',
            'timestamp': datetime.now().isoformat(),
            'device_info': {
                'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'peak_flops': flops_data['efficiency_metrics']['device_peak_flops'],
                'peak_memory_bandwidth': intensity_data.get('device_memory_bandwidth_gb_per_sec', 0) * 1e9
            },
            'performance_point': {
                'arithmetic_intensity': intensity_data.get('arithmetic_intensity_flops_per_byte', 0),
                'achieved_performance': flops_data['performance_metrics']['flops_per_sec'],
                'mfu_percent': flops_data['efficiency_metrics']['mfu_percent']
            },
            'optimization_targets': self._generate_optimization_targets(flops_data, intensity_data)
        }

        # Save roofline data
        roofline_path = output_path / "roofline_data.json"
        with open(roofline_path, 'w') as f:
            json.dump(roofline_data, f, indent=2)

        print(f"Roofline data generated: {roofline_path}")
        return str(roofline_path)

    def _generate_optimization_targets(self, flops_data: Dict, intensity_data: Dict) -> List[Dict[str, str]]:
        """Generate optimization targets based on analysis."""
        targets = []

        # MFU-based recommendations
        mfu = flops_data['efficiency_metrics']['mfu_percent']
        if mfu < 30:
            targets.append({
                'target': 'Kernel Fusion',
                'reason': f'Low MFU ({mfu:.1f}%) indicates kernel launch overhead',
                'expected_improvement': '2-3x speedup potential'
            })

        # Arithmetic intensity recommendations
        ai = intensity_data.get('arithmetic_intensity_flops_per_byte', 0)
        if ai < 10:
            targets.append({
                'target': 'Memory Optimization',
                'reason': f'Low arithmetic intensity ({ai:.2f}) indicates memory-bound operations',
                'expected_improvement': 'Flash Attention, gradient checkpointing'
            })

        # Model-specific optimizations
        targets.extend([
            {
                'target': 'QKV Fusion',
                'reason': 'Separate linear projections create multiple kernel launches',
                'expected_improvement': '20-30% reduction in attention overhead'
            },
            {
                'target': 'SwiGLU Fusion',
                'reason': 'Gate and up projections can be computed together',
                'expected_improvement': '15-25% FFN speedup'
            }
        ])

        return targets


def main():
    """Main entry point for DeepSpeed FLOPS analysis."""
    parser = argparse.ArgumentParser(description='DeepSpeed FLOPS Profiler for Tiny LLaMA V1')

    # Model configuration
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for profiling')
    parser.add_argument('--seq-len', type=int, default=128, help='Sequence length')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of layers')

    # Profiling configuration
    parser.add_argument('--num-steps', type=int, default=10, help='Number of profiling steps')
    parser.add_argument('--output-dir', type=str, default='./flops_analysis', help='Output directory')
    parser.add_argument('--detailed-analysis', action='store_true', help='Enable detailed FLOPS breakdown')

    # Analysis options
    parser.add_argument('--analyze-results', type=str, help='Analyze existing FLOPS results file')
    parser.add_argument('--generate-roofline', action='store_true', help='Generate roofline analysis data')
    parser.add_argument('--computational-intensity', action='store_true', help='Analyze computational intensity')

    args = parser.parse_args()

    if not DEEPSPEED_AVAILABLE and not args.analyze_results:
        print("DeepSpeed not available. Please install DeepSpeed for FLOPS profiling.")
        print("   pip install deepspeed")
        return

    # Create analyzer
    analyzer = FLOPSAnalyzer(args.output_dir)

    print("DEEPSPEED FLOPS PROFILER - TINY LLAMA V1")
    print("=" * 60)

    try:
        # Analyze existing results
        if args.analyze_results:
            with open(args.analyze_results, 'r') as f:
                flops_data = json.load(f)
            print(f"📁 Analyzing existing results: {args.analyze_results}")
            # Analysis logic here
            return

        # Run new FLOPS profiling
        config = TinyLlamaConfig(
            hidden_dim=args.hidden_dim,
            n_layers=args.num_layers,
            max_seq_len=args.seq_len
        )

        # Profile FLOPS
        flops_results = analyzer.profile_model_flops(
            config=config,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            detailed_analysis=args.detailed_analysis
        )

        if 'error' in flops_results:
            print(f"FLOPS profiling failed: {flops_results['error']}")
            return

        # Computational intensity analysis
        if args.computational_intensity:
            intensity_results = analyzer.analyze_computational_intensity(flops_results)
            if 'error' not in intensity_results:
                print("Computational intensity analysis completed")

        # Generate roofline data
        if args.generate_roofline:
            roofline_path = analyzer.generate_roofline_data(args.output_dir)
            print(f"Roofline data generated: {roofline_path}")

        print(f"\nFLOPS analysis completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"\nKey Metrics:")
        print(f"   Model FLOPS Utilization: {flops_results['efficiency_metrics']['mfu_percent']:.1f}%")
        print(f"   Throughput: {flops_results['performance_metrics']['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"   FLOPS per parameter: {flops_results['flops_analysis']['flops_per_parameter']:.2f}")

    except Exception as e:
        print(f"FAIL Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()