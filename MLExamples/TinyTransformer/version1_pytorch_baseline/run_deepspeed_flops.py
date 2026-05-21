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

        # Run profiling.
        #
        # Important caveats addressed in this loop:
        #   * DeepSpeed's `FlopsProfiler` instruments forward hooks only --
        #     `loss.backward()` is NOT counted. We therefore label these as
        #     "forward FLOPS" and separately estimate fwd+bwd as ~3x forward
        #     (the conventional 1 fwd + 2 bwd transformer-training factor).
        #   * `prof.get_total_flops()` returns a CUMULATIVE counter since the
        #     last `start_profile()`. To get a true per-step value we restart
        #     the profile each step (start_profile / stop_profile / end_profile),
        #     read the per-step total, then start again for the next step.
        #   * We also track a separate "steady-state" average that drops the
        #     first step (lazy CUDA init / kernel JIT dominates step 0).
        model.train()

        total_fwd_flops = 0
        total_time = 0
        step_results = []
        detailed_profile_text = None

        # We can call print_model_profile only between start_profile and
        # end_profile. Capture it from the second step (steady-state, after
        # warm-up) to avoid lazy-init artefacts.
        capture_breakdown_at_step = 1 if num_steps > 1 else 0

        # `end_profile()` (if present) tears down hooks; `stop_profile()` only
        # freezes counters. We use start_profile + (optional print) +
        # end_profile each step so per-step FLOPS = the value read this step.
        has_end_profile = hasattr(prof, 'end_profile')

        for step in range(num_steps):
            # Get batch
            input_ids, labels = dataset.get_batch(batch_size)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Fresh profiling window for this step.
            prof.start_profile()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            # Forward pass (this is what DeepSpeed counts).
            outputs = model(input_ids, labels)
            loss = outputs['loss']

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_time = time.time() - start_time

            # Read per-step forward FLOPS while the profile is still active.
            if hasattr(prof, 'get_total_flops'):
                try:
                    step_fwd_flops = float(prof.get_total_flops())
                except Exception as e:
                    print(f"   Warning: get_total_flops failed at step {step}: {e}")
                    step_fwd_flops = float(self._estimate_transformer_flops(config, batch_size))
            else:
                step_fwd_flops = float(self._estimate_transformer_flops(config, batch_size))

            # Capture the per-module breakdown once, on a steady-state step,
            # while the profile is still active.
            if (
                detailed_analysis
                and detailed_profile_text is None
                and step == capture_breakdown_at_step
                and hasattr(prof, 'print_model_profile')
            ):
                try:
                    import io, contextlib
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        # `profile_step` is just a label DeepSpeed prints in
                        # the report header; depth=-1 means show full tree.
                        prof.print_model_profile(
                            profile_step=step,
                            module_depth=-1,
                            top_modules=50,
                        )
                    detailed_profile_text = buf.getvalue()
                    # Persist the human-readable breakdown next to the JSON.
                    breakdown_path = self.output_dir / "model_profile.txt"
                    with open(breakdown_path, 'w') as f:
                        f.write(detailed_profile_text)
                    print(f"   Per-module FLOPS breakdown written to: {breakdown_path}")
                except Exception as e:
                    print(f"   Warning: print_model_profile failed: {e}")
                    detailed_profile_text = f"Profile generation failed: {e}"

            # Tear the profile down for this step so the next start_profile()
            # gives us a fresh, non-cumulative counter.
            if has_end_profile:
                try:
                    prof.end_profile()
                except Exception:
                    # Older DeepSpeed: stop_profile() is the only option.
                    prof.stop_profile()
            else:
                prof.stop_profile()

            # Backward pass: NOT counted by DeepSpeed FlopsProfiler.
            # Time it separately so the user can see fwd vs bwd cost.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            bwd_start = time.time()
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_time = time.time() - bwd_start
            step_time = forward_time + backward_time

            total_fwd_flops += step_fwd_flops
            total_time += step_time

            step_results.append({
                'step': step,
                'loss': loss.item(),
                'forward_flops': step_fwd_flops,
                'forward_time': forward_time,
                'backward_time': backward_time,
                'step_time': step_time,
                'forward_flops_per_sec':
                    step_fwd_flops / forward_time if forward_time > 0 else 0,
            })

            if step % 2 == 0:
                print(f"   Step {step}: Loss {loss.item():.4f}, "
                      f"fwd FLOPS {step_fwd_flops:.2e}, "
                      f"fwd {forward_time*1000:.1f}ms, "
                      f"bwd {backward_time*1000:.1f}ms")

            # Clear gradients for next step
            model.zero_grad()

        # Aggregate.
        avg_time_per_step = total_time / num_steps if num_steps > 0 else 0
        avg_fwd_flops_per_step = total_fwd_flops / num_steps if num_steps > 0 else 0
        # Conventional transformer-training rule of thumb: backward ~ 2x forward.
        BWD_FACTOR = 2.0
        avg_total_flops_per_step_estimate = avg_fwd_flops_per_step * (1.0 + BWD_FACTOR)
        throughput = batch_size / avg_time_per_step if avg_time_per_step > 0 else 0

        # Steady-state numbers: drop the first step to remove lazy-init outlier.
        steady = step_results[1:] if len(step_results) > 1 else step_results
        steady_avg_step_time = (
            sum(r['step_time'] for r in steady) / len(steady) if steady else 0.0
        )
        steady_avg_fwd_flops = (
            sum(r['forward_flops'] for r in steady) / len(steady) if steady else 0.0
        )
        steady_throughput = (
            batch_size / steady_avg_step_time if steady_avg_step_time > 0 else 0.0
        )

        # MFU: forward-only and a fwd+bwd-estimate variant. Use steady-state
        # timings so the lazy-init step doesn't poison the result.
        peak_flops = self._get_device_peak_flops()
        mfu_fwd = self._calculate_mfu(
            model_flops=steady_avg_fwd_flops,
            # Forward only: use forward time, not full step time.
            time_per_step=(
                sum(r['forward_time'] for r in steady) / len(steady)
                if steady else 0.0
            ),
            device_peak_flops=peak_flops,
        )
        mfu_total_estimate = self._calculate_mfu(
            model_flops=steady_avg_fwd_flops * (1.0 + BWD_FACTOR),
            time_per_step=steady_avg_step_time,
            device_peak_flops=peak_flops,
        )

        flops_summary_total = total_fwd_flops
        try:
            params_summary = prof.get_total_params() if hasattr(prof, 'get_total_params') else total_params
        except Exception:
            params_summary = total_params

        results = {
            'model_info': {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'deepspeed_total_params': params_summary,
                'config': config.to_dict(),
            },
            'profiling_config': {
                'batch_size': batch_size,
                'sequence_length': config.max_seq_len,
                'num_steps': num_steps,
                'device': str(device),
            },
            'flops_analysis': {
                'note': (
                    "DeepSpeed's FlopsProfiler counts FORWARD-only FLOPS via "
                    "module forward hooks. 'avg_total_flops_per_step_estimate' "
                    "applies the conventional ~3x (1 fwd + 2 bwd) transformer "
                    "training factor and is therefore an estimate, not a "
                    "measurement. Per-step values are true per-step deltas "
                    "(start_profile / end_profile each step), not cumulative."
                ),
                'total_forward_flops_summed': flops_summary_total,
                'avg_forward_flops_per_step': avg_fwd_flops_per_step,
                'avg_total_flops_per_step_estimate':
                    avg_total_flops_per_step_estimate,
                'fwd_plus_bwd_estimation_factor': 1.0 + BWD_FACTOR,
                'forward_flops_per_parameter':
                    avg_fwd_flops_per_step / max(1, total_params),
                'detailed_profile':
                    detailed_profile_text or "Detailed profile not available",
            },
            'performance_metrics': {
                'avg_time_per_step': avg_time_per_step,
                'avg_forward_time_per_step':
                    sum(r['forward_time'] for r in step_results) / num_steps
                    if num_steps else 0.0,
                'avg_backward_time_per_step':
                    sum(r['backward_time'] for r in step_results) / num_steps
                    if num_steps else 0.0,
                'throughput_samples_per_sec': throughput,
                'avg_loss': float(np.mean([r['loss'] for r in step_results])) if step_results else 0.0,
                'forward_flops_per_sec':
                    avg_fwd_flops_per_step / avg_time_per_step
                    if avg_time_per_step > 0 else 0,
            },
            'steady_state_metrics': {
                'note': (
                    "Steady-state metrics exclude step 0 to drop the lazy CUDA "
                    "init / kernel JIT outlier."
                ),
                'num_steady_steps': len(steady),
                'avg_step_time': steady_avg_step_time,
                'avg_forward_flops_per_step': steady_avg_fwd_flops,
                'throughput_samples_per_sec': steady_throughput,
            },
            'efficiency_metrics': {
                'note': (
                    "mfu_forward = forward FLOPS / forward time vs peak "
                    "(measurement-based, but ignores backward). "
                    "mfu_total_estimate = (1+2)*forward FLOPS / step time vs "
                    "peak (uses the 3x rule of thumb for backward; estimate). "
                    "Both use steady-state timings (step 0 excluded)."
                ),
                'device_peak_flops_fp32': peak_flops,
                'mfu_forward_steady_state': mfu_fwd,
                'mfu_total_estimate_steady_state': mfu_total_estimate,
            },
            'step_by_step_results': step_results,
            'timestamp': datetime.now().isoformat(),
        }

        # Save results
        results_path = self.output_dir / "flops_profile.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nFLOPS Analysis Summary (forward-only FLOPS unless stated):")
        print(f"   Avg forward FLOPS / step:        {avg_fwd_flops_per_step:.3e}")
        print(f"   Estimated fwd+bwd FLOPS / step:  "
              f"{avg_total_flops_per_step_estimate:.3e} (= 3x forward)")
        print(f"   Avg fwd time / step:             "
              f"{results['performance_metrics']['avg_forward_time_per_step']*1000:.2f} ms")
        print(f"   Avg bwd time / step:             "
              f"{results['performance_metrics']['avg_backward_time_per_step']*1000:.2f} ms")
        print(f"   Throughput (all steps):          {throughput:.1f} samples/sec")
        print(f"   Throughput (steady state):       {steady_throughput:.1f} samples/sec")
        print(f"   MFU forward (steady, measured):  "
              f"{mfu_fwd['mfu_percent']:.3f}%")
        print(f"   MFU fwd+bwd (steady, ~3x est.):  "
              f"{mfu_total_estimate['mfu_percent']:.3f}%")
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

        # Arithmetic intensity (FLOPS per byte). Uses the fwd+bwd ~3x estimate
        # so the figure represents a full training step.
        flops_block = flops_data['flops_analysis']
        avg_flops = flops_block['avg_total_flops_per_step_estimate']
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
                'peak_flops': flops_data['efficiency_metrics']['device_peak_flops_fp32'],
                'peak_memory_bandwidth': device_memory_bandwidth,
                'achieved_flops': perf_metrics['forward_flops_per_sec'],
                'achieved_bandwidth': memory_bandwidth_used,
            },
            'note': (
                "Arithmetic intensity / bandwidth here are ANALYTICAL "
                "estimates: bytes = params*4 + 3*activations (rough), and "
                "FLOPS uses the fwd+bwd ~3x estimate. Real values require "
                "rocprof/omniperf HW counters."
            ),
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
                'peak_flops': flops_data['efficiency_metrics']['device_peak_flops_fp32'],
                'peak_memory_bandwidth': intensity_data.get('device_memory_bandwidth_gb_per_sec', 0) * 1e9
            },
            'performance_point': {
                'arithmetic_intensity': intensity_data.get('arithmetic_intensity_flops_per_byte', 0),
                'achieved_performance': flops_data['performance_metrics']['forward_flops_per_sec'],
                'mfu_percent':
                    flops_data['efficiency_metrics']['mfu_forward_steady_state']['mfu_percent'],
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
        mfu = flops_data['efficiency_metrics']['mfu_forward_steady_state']['mfu_percent']
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
            results_path = Path(args.analyze_results)
            # If the user passed a bare filename, also look inside --output-dir.
            if not results_path.exists():
                fallback = Path(args.output_dir) / results_path.name
                if fallback.exists():
                    print(f"Note: '{results_path}' not found, using '{fallback}' instead.")
                    results_path = fallback
                else:
                    print(f"Results file not found: {args.analyze_results}")
                    print(f"   Also looked in: {fallback}")
                    return

            with open(results_path, 'r') as f:
                flops_data = json.load(f)
            print(f"Analyzing existing results: {results_path}")

            model_info = flops_data.get('model_info', {})
            cfg = model_info.get('config', {})
            pcfg = flops_data.get('profiling_config', {})
            flops = flops_data.get('flops_analysis', {})
            perf = flops_data.get('performance_metrics', {})
            eff = flops_data.get('efficiency_metrics', {})
            steps = flops_data.get('step_by_step_results', [])

            print("\nModel:")
            print(f"   Params: {model_info.get('total_params', 0):,}")
            print(f"   hidden_dim={cfg.get('hidden_dim')}, n_layers={cfg.get('n_layers')}, "
                  f"n_heads={cfg.get('n_heads')}, max_seq_len={cfg.get('max_seq_len')}")
            print(f"   Profiling: batch_size={pcfg.get('batch_size')}, "
                  f"seq_len={pcfg.get('sequence_length')}, "
                  f"num_steps={pcfg.get('num_steps')}, device={pcfg.get('device')}")

            steady_block = flops_data['steady_state_metrics']

            print("\nFLOPS:")
            print(f"   Avg forward FLOPS / step:        "
                  f"{flops['avg_forward_flops_per_step']:.3e}")
            print(f"   Estimated fwd+bwd FLOPS / step:  "
                  f"{flops['avg_total_flops_per_step_estimate']:.3e}  "
                  f"(factor = {flops['fwd_plus_bwd_estimation_factor']:.1f}x forward; "
                  f"backward is NOT measured by DeepSpeed)")
            print(f"   Forward FLOPS / parameter:       "
                  f"{flops['forward_flops_per_parameter']:.2f}")

            print("\nPerformance:")
            print(f"   Avg step time:        {perf['avg_time_per_step']*1000:.2f} ms")
            print(f"   Avg forward time:     "
                  f"{perf['avg_forward_time_per_step']*1000:.2f} ms")
            print(f"   Avg backward time:    "
                  f"{perf['avg_backward_time_per_step']*1000:.2f} ms")
            print(f"   Throughput (all):     {perf['throughput_samples_per_sec']:.2f} samples/sec")
            print(f"   Throughput (steady):  "
                  f"{steady_block['throughput_samples_per_sec']:.2f} samples/sec "
                  f"(over {steady_block['num_steady_steps']} steady-state steps)")
            print(f"   Achieved fwd FLOPS/s: {perf['forward_flops_per_sec']:.3e}")
            print(f"   Avg loss:             {perf['avg_loss']:.4f}")

            print("\nEfficiency:")
            peak = eff['device_peak_flops_fp32']
            print(f"   Device peak FLOPS/s (FP32): {peak:.3e}")
            print(f"   MFU forward (steady, measured):  "
                  f"{eff['mfu_forward_steady_state']['mfu_percent']:.3f}%")
            print(f"   MFU fwd+bwd (steady, ~3x est.):  "
                  f"{eff['mfu_total_estimate_steady_state']['mfu_percent']:.3f}%")

            print("\nPer-step (true per-step FLOPS, fwd-only):")
            for i, s in enumerate(steps):
                print(f"   step {s.get('step', i):>2}: "
                      f"fwd {s['forward_flops']:.3e} FLOPS, "
                      f"fwd {s['forward_time']*1000:>7.2f} ms, "
                      f"bwd {s['backward_time']*1000:>7.2f} ms, "
                      f"loss {s['loss']:.4f}")

            # Persist a compact summary next to the input file.
            summary_path = results_path.with_name(results_path.stem + ".analysis.json")
            summary = {
                'source': str(results_path),
                'model_params': model_info.get('total_params', 0),
                'config': cfg,
                'profiling_config': pcfg,
                'reported': {
                    'avg_forward_flops_per_step':
                        flops['avg_forward_flops_per_step'],
                    'avg_total_flops_per_step_estimate':
                        flops['avg_total_flops_per_step_estimate'],
                    'avg_forward_time_per_step_ms':
                        perf['avg_forward_time_per_step'] * 1000,
                    'avg_backward_time_per_step_ms':
                        perf['avg_backward_time_per_step'] * 1000,
                    'throughput_samples_per_sec':
                        perf['throughput_samples_per_sec'],
                    'steady_state': steady_block,
                    'mfu_forward_steady_state':
                        eff['mfu_forward_steady_state']['mfu_percent'],
                    'mfu_total_estimate_steady_state':
                        eff['mfu_total_estimate_steady_state']['mfu_percent'],
                },
            }

            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary written to: {summary_path}")
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
        print(f"Results saved to: {args.output_dir}")
        print(f"\nKey Metrics (forward-only unless stated):")
        eff = flops_results['efficiency_metrics']
        flops_block = flops_results['flops_analysis']
        steady = flops_results.get('steady_state_metrics', {})
        fwd_mfu = eff.get('mfu_forward_steady_state', {}).get('mfu_percent', 0)
        tot_mfu = eff.get('mfu_total_estimate_steady_state', {}).get('mfu_percent', 0)
        print(f"   MFU forward (steady, measured):       {fwd_mfu:.3f}%")
        print(f"   MFU fwd+bwd (steady, ~3x estimate):   {tot_mfu:.3f}%")
        print(f"   Throughput (steady):                  "
              f"{steady.get('throughput_samples_per_sec', 0):.1f} samples/sec")
        print(f"   Forward FLOPS per parameter:          "
              f"{flops_block.get('forward_flops_per_parameter', 0):.2f}")

    except Exception as e:
        print(f"FAIL Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()