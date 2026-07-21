#!/usr/bin/env python3
"""
DeepSpeed FLOPS Profiler Integration for Tiny OpenFold V1

This script provides comprehensive FLOPS analysis using DeepSpeed's FLOPS profiler
to measure computational efficiency and identify optimization opportunities for 
the Evoformer architecture.

Features:
- Detailed FLOPS breakdown by operation type (MSA attention, pair updates, triangle mult)
- Model FLOPS Utilization (MFU) calculation
- Computational intensity analysis
- Memory bandwidth requirements
- Arithmetic intensity metrics
- Roofline model preparation data

Usage:
    # Run FLOPS profiling with default settings
    python run_deepspeed_flops.py

    # Custom configuration
    python run_deepspeed_flops.py --batch-size 4 --seq-len 64

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

# Import the model from tiny_openfold_v1
from tiny_openfold_v1 import (
    TinyOpenFold, 
    TinyOpenFoldConfig, 
    ProteinDataset,
    setup_deterministic_environment
)

# Optional DeepSpeed import
try:
    from deepspeed.profiling.flops_profiler import FlopsProfiler
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False


class EvoformerFLOPSAnalyzer:
    """Comprehensive FLOPS analysis for Evoformer architecture."""

    def __init__(self, output_dir: str = "./flops_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_results = {}

    def profile_model_flops(
        self,
        config: TinyOpenFoldConfig,
        batch_size: int = 4,
        num_steps: int = 10,
        detailed_analysis: bool = True,
        device_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Profile model FLOPS using DeepSpeed profiler."""

        if not DEEPSPEED_AVAILABLE:
            return {'error': 'DeepSpeed not available for FLOPS profiling'}

        print(f"Starting FLOPS Analysis - Evoformer Architecture")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Batch size: {batch_size}")
        print(f"   Sequence length: {config.max_seq_len}")
        print(f"   MSA sequences: {config.n_seqs}")
        print(f"   Analysis steps: {num_steps}")

        # Setup environment
        setup_deterministic_environment()
        
        # Device selection
        if device_id is not None:
            if not torch.cuda.is_available():
                print(f"   Warning: CUDA not available, ignoring device_id={device_id}")
                device = torch.device("cpu")
            elif device_id >= torch.cuda.device_count():
                raise ValueError(f"Device {device_id} not available. Only {torch.cuda.device_count()} GPU(s) found.")
            else:
                device = torch.device(f"cuda:{device_id}")
                print(f"   Using GPU: {device_id}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model and dataset
        model = TinyOpenFold(config).to(device)
        dataset = ProteinDataset(config)

        # Initialize FLOPS profiler
        prof = FlopsProfiler(model)

        # Model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nModel Information:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size (FP32): {total_params * 4 / 1e6:.1f} MB")
        print(f"   Evoformer blocks: {config.n_evoformer_blocks}")
        print(f"   MSA dimension: {config.msa_dim}")
        print(f"   Pair dimension: {config.pair_dim}")

        # Run profiling
        model.train()
        prof.start_profile()

        total_flops = 0
        total_time = 0
        step_results = []

        for step in range(num_steps):
            # Get batch
            msa_tokens, pair_features, target_distances = dataset.get_batch(batch_size)
            msa_tokens = msa_tokens.to(device)
            pair_features = pair_features.to(device)
            target_distances = target_distances.to(device)

            # Time the forward pass
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Forward pass
            outputs = model(msa_tokens, pair_features, target_distances)
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
                step_flops = self._estimate_evoformer_flops(config, batch_size)

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

        # Evoformer-specific FLOPS breakdown
        evoformer_breakdown = self._estimate_evoformer_breakdown(config, batch_size)

        results = {
            'model_info': {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'config': config.to_dict(),
                'architecture': 'Evoformer'
            },
            'profiling_config': {
                'batch_size': batch_size,
                'sequence_length': config.max_seq_len,
                'msa_sequences': config.n_seqs,
                'num_steps': num_steps,
                'device': str(device)
            },
            'flops_analysis': {
                'total_flops': flops_summary,
                'avg_flops_per_step': avg_flops_per_step,
                'flops_per_parameter': avg_flops_per_step / max(1, total_params),
                'evoformer_breakdown': evoformer_breakdown,
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
        
        print(f"\nEvoformer FLOPS Breakdown:")
        for component, flops in evoformer_breakdown.items():
            pct = (flops / avg_flops_per_step * 100) if avg_flops_per_step > 0 else 0
            print(f"   {component}: {flops:.2e} ({pct:.1f}%)")
        
        print(f"\n   Results saved to: {results_path}")

        return results

    def profile_multi_gpu_flops(
        self,
        config: TinyOpenFoldConfig,
        batch_size: int = 4,
        num_steps: int = 10,
        device_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Profile FLOPS across multiple GPUs for comparative analysis."""
        
        print(f"\nStarting Multi-GPU FLOPS Analysis - Evoformer Architecture")
        print(f"   Output directory: {self.output_dir}")
        
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available for multi-GPU profiling'}
        
        # Determine which GPUs to use
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        else:
            # Validate device IDs
            for dev_id in device_ids:
                if dev_id >= torch.cuda.device_count():
                    raise ValueError(f"Device {dev_id} not available. Only {torch.cuda.device_count()} GPU(s) found.")
        
        num_gpus = len(device_ids)
        print(f"   Profiling on {num_gpus} GPU(s): {device_ids}")
        print(f"   Batch size per GPU: {batch_size}")
        print(f"   Total effective batch size: {batch_size * num_gpus}")
        print(f"   Sequence length: {config.max_seq_len}")
        print(f"   Analysis steps: {num_steps}")
        
        # Profile each GPU individually
        per_gpu_results = {}
        
        for gpu_id in device_ids:
            print(f"\n{'='*70}")
            print(f"Profiling GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            print(f"{'='*70}")
            
            # Profile this GPU
            gpu_results = self.profile_model_flops(
                config=config,
                batch_size=batch_size,
                num_steps=num_steps,
                detailed_analysis=False,
                device_id=gpu_id
            )
            
            per_gpu_results[f"gpu_{gpu_id}"] = gpu_results
            
            # Print summary for this GPU
            if 'error' not in gpu_results:
                print(f"\n   GPU {gpu_id} Summary:")
                print(f"      MFU: {gpu_results['efficiency_metrics']['mfu_percent']:.1f}%")
                print(f"      Achieved TFLOPS: {gpu_results['efficiency_metrics']['achieved_tflops']:.2f}")
                print(f"      Throughput: {gpu_results['performance_metrics']['throughput_samples_per_sec']:.1f} samples/sec")
        
        # Aggregate results
        print(f"\n{'='*70}")
        print(f"Multi-GPU Aggregate Analysis")
        print(f"{'='*70}")
        
        aggregate_results = self._aggregate_multi_gpu_results(
            per_gpu_results, 
            device_ids, 
            config, 
            batch_size, 
            num_steps
        )
        
        # Save multi-GPU results
        multi_gpu_path = self.output_dir / "flops_profile_multi_gpu.json"
        with open(multi_gpu_path, 'w') as f:
            json.dump(aggregate_results, f, indent=2)
        
        print(f"\n   Multi-GPU results saved to: {multi_gpu_path}")
        
        # Print aggregate summary
        print(f"\nAggregate Multi-GPU Summary:")
        print(f"   Number of GPUs: {num_gpus}")
        print(f"   Total System TFLOPS: {aggregate_results['aggregate_metrics']['total_system_tflops']:.2f}")
        print(f"   Average MFU: {aggregate_results['aggregate_metrics']['avg_mfu_percent']:.1f}%")
        print(f"   Total Throughput: {aggregate_results['aggregate_metrics']['total_throughput']:.1f} samples/sec")
        print(f"   Multi-GPU Efficiency: {aggregate_results['aggregate_metrics']['multi_gpu_efficiency_percent']:.1f}%")
        
        return aggregate_results

    def _aggregate_multi_gpu_results(
        self, 
        per_gpu_results: Dict[str, Dict], 
        device_ids: List[int],
        config: TinyOpenFoldConfig,
        batch_size: int,
        num_steps: int
    ) -> Dict[str, Any]:
        """Aggregate results from multiple GPU profiling runs."""
        
        num_gpus = len(device_ids)
        
        # Collect metrics from each GPU
        mfu_values = []
        achieved_tflops = []
        throughput_values = []
        avg_time_per_step = []
        
        for gpu_id in device_ids:
            gpu_key = f"gpu_{gpu_id}"
            if gpu_key in per_gpu_results and 'error' not in per_gpu_results[gpu_key]:
                result = per_gpu_results[gpu_key]
                mfu_values.append(result['efficiency_metrics']['mfu_percent'])
                achieved_tflops.append(result['efficiency_metrics']['achieved_tflops'])
                throughput_values.append(result['performance_metrics']['throughput_samples_per_sec'])
                avg_time_per_step.append(result['performance_metrics']['avg_time_per_step'])
        
        # Calculate aggregate metrics
        avg_mfu = np.mean(mfu_values) if mfu_values else 0
        total_tflops = sum(achieved_tflops)
        total_throughput = sum(throughput_values)
        avg_time = np.mean(avg_time_per_step) if avg_time_per_step else 0
        
        # Calculate multi-GPU efficiency (ideal = 100% means linear scaling)
        # Efficiency = (Total Throughput) / (Single GPU Throughput × N)
        if len(throughput_values) > 0:
            single_gpu_throughput = throughput_values[0] if throughput_values else 0
            ideal_throughput = single_gpu_throughput * num_gpus
            multi_gpu_efficiency = (total_throughput / ideal_throughput * 100) if ideal_throughput > 0 else 0
        else:
            multi_gpu_efficiency = 0
        
        # Get device information
        device_info = []
        for gpu_id in device_ids:
            device_info.append({
                'gpu_id': gpu_id,
                'name': torch.cuda.get_device_name(gpu_id),
                'mfu_percent': mfu_values[device_ids.index(gpu_id)] if gpu_id < len(mfu_values) else 0,
                'achieved_tflops': achieved_tflops[device_ids.index(gpu_id)] if gpu_id < len(achieved_tflops) else 0,
                'throughput': throughput_values[device_ids.index(gpu_id)] if gpu_id < len(throughput_values) else 0
            })
        
        aggregate_results = {
            'multi_gpu_config': {
                'num_gpus': num_gpus,
                'device_ids': device_ids,
                'batch_size_per_gpu': batch_size,
                'total_batch_size': batch_size * num_gpus,
                'num_steps': num_steps
            },
            'model_config': config.to_dict(),
            'per_gpu_results': per_gpu_results,
            'device_info': device_info,
            'aggregate_metrics': {
                'avg_mfu_percent': avg_mfu,
                'mfu_std_dev': np.std(mfu_values) if len(mfu_values) > 1 else 0,
                'total_system_tflops': total_tflops,
                'avg_tflops_per_gpu': np.mean(achieved_tflops) if achieved_tflops else 0,
                'total_throughput': total_throughput,
                'avg_throughput_per_gpu': np.mean(throughput_values) if throughput_values else 0,
                'avg_time_per_step': avg_time,
                'multi_gpu_efficiency_percent': multi_gpu_efficiency,
                'scaling_efficiency': {
                    'ideal_speedup': num_gpus,
                    'actual_speedup': (throughput_values[0] * num_gpus / total_throughput) if total_throughput > 0 and throughput_values else 0,
                    'efficiency_ratio': multi_gpu_efficiency / 100
                }
            },
            'comparison': {
                'single_gpu_throughput': throughput_values[0] if throughput_values else 0,
                'multi_gpu_throughput': total_throughput,
                'speedup': total_throughput / throughput_values[0] if throughput_values and throughput_values[0] > 0 else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return aggregate_results

    def _estimate_evoformer_flops(self, config: TinyOpenFoldConfig, batch_size: int) -> float:
        """Estimate FLOPS for Evoformer model (fallback if DeepSpeed fails)."""
        B = batch_size
        L = config.max_seq_len
        N = config.n_seqs
        d_msa = config.msa_dim
        d_pair = config.pair_dim
        n_blocks = config.n_evoformer_blocks
        n_heads_msa = config.n_heads_msa
        n_heads_pair = config.n_heads_pair
        d_msa_inter = config.msa_intermediate_dim
        d_pair_inter = config.pair_intermediate_dim

        # Embedding FLOPS (input projection)
        # MSA embedding: B * N * L * vocab_size * d_msa
        embed_flops = B * N * L * config.vocab_size * d_msa
        # Pair embedding: B * L * L * pair_input_dim * d_pair
        embed_flops += B * L * L * config.pair_input_dim * d_pair

        # Per Evoformer block FLOPS
        block_flops = 0

        # === MSA STACK ===
        # MSA Row Attention
        # Q, K, V projections: 3 * B * N * L * d_msa * d_msa
        msa_qkv_flops = 3 * B * N * L * d_msa * d_msa
        # Attention scores: B * N * n_heads_msa * L * L * (d_msa / n_heads_msa)
        msa_attn_scores = B * N * n_heads_msa * L * L * (d_msa // n_heads_msa)
        # Attention output: B * N * n_heads_msa * L * (d_msa / n_heads_msa) * L
        msa_attn_out = B * N * n_heads_msa * L * (d_msa // n_heads_msa) * L
        # Output projection: B * N * L * d_msa * d_msa
        msa_out_proj = B * N * L * d_msa * d_msa
        
        msa_row_attn = msa_qkv_flops + msa_attn_scores + msa_attn_out + msa_out_proj

        # MSA Column Attention (similar to row but different dimension)
        msa_col_attn = msa_row_attn  # Approximation

        # MSA Transition (FFN)
        # Linear 1: B * N * L * d_msa * d_msa_inter
        # Linear 2: B * N * L * d_msa_inter * d_msa
        msa_transition = B * N * L * d_msa * d_msa_inter + B * N * L * d_msa_inter * d_msa

        # Outer Product Mean
        # Projects MSA to create pair update
        # B * L * L * N * d_msa * outer_product_dim
        outer_product = B * L * L * N * d_msa * config.outer_product_dim

        msa_stack_total = msa_row_attn + msa_col_attn + msa_transition + outer_product

        # === PAIR STACK ===
        # Triangle Multiplication Outgoing
        # 3 projections + matmul: estimate as 4 * B * L * L * d_pair * d_pair
        triangle_mult_out = 4 * B * L * L * d_pair * d_pair

        # Triangle Multiplication Incoming
        triangle_mult_in = 4 * B * L * L * d_pair * d_pair

        # Triangle Attention Starting/Ending (simplified)
        # Similar to standard attention but on pairs
        triangle_attn = 2 * (4 * B * L * L * d_pair * d_pair)

        # Pair Transition (FFN)
        pair_transition = B * L * L * d_pair * d_pair_inter + B * L * L * d_pair_inter * d_pair

        pair_stack_total = triangle_mult_out + triangle_mult_in + triangle_attn + pair_transition

        # Layer normalization (relatively small, but included for completeness)
        # Multiple layer norms throughout: ~10 per block * B * N * L * d_msa (rough estimate)
        layernorm_flops = 10 * B * N * L * d_msa

        block_flops = msa_stack_total + pair_stack_total + layernorm_flops

        # Total model FLOPS
        total_flops = embed_flops + (n_blocks * block_flops)

        # Output head (distance prediction)
        # B * L * L * d_pair * num_distance_bins (simplified)
        output_flops = B * L * L * d_pair * 64  # Assuming 64 distance bins
        total_flops += output_flops

        return total_flops

    def _estimate_evoformer_breakdown(self, config: TinyOpenFoldConfig, batch_size: int) -> Dict[str, float]:
        """Provide detailed breakdown of FLOPS by Evoformer component."""
        B = batch_size
        L = config.max_seq_len
        N = config.n_seqs
        d_msa = config.msa_dim
        d_pair = config.pair_dim
        n_blocks = config.n_evoformer_blocks

        breakdown = {}

        # MSA Row/Column Attention
        msa_attn_per_block = 2 * (4 * B * N * L * d_msa * d_msa + B * N * config.n_heads_msa * L * L * (d_msa // config.n_heads_msa))
        breakdown['msa_attention'] = msa_attn_per_block * n_blocks

        # MSA Transition
        msa_transition_per_block = B * N * L * d_msa * config.msa_intermediate_dim + B * N * L * config.msa_intermediate_dim * d_msa
        breakdown['msa_transition'] = msa_transition_per_block * n_blocks

        # Outer Product Mean
        outer_product_per_block = B * L * L * N * d_msa * config.outer_product_dim
        breakdown['outer_product_mean'] = outer_product_per_block * n_blocks

        # Triangle Multiplication
        triangle_mult_per_block = 8 * B * L * L * d_pair * d_pair
        breakdown['triangle_multiplication'] = triangle_mult_per_block * n_blocks

        # Triangle Attention
        triangle_attn_per_block = 8 * B * L * L * d_pair * d_pair
        breakdown['triangle_attention'] = triangle_attn_per_block * n_blocks

        # Pair Transition
        pair_transition_per_block = B * L * L * d_pair * config.pair_intermediate_dim + B * L * L * config.pair_intermediate_dim * d_pair
        breakdown['pair_transition'] = pair_transition_per_block * n_blocks

        # Embeddings
        breakdown['embeddings'] = B * N * L * config.vocab_size * d_msa + B * L * L * config.pair_input_dim * d_pair

        # Output head
        breakdown['output_head'] = B * L * L * d_pair * 64

        return breakdown

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
            'mi300': 61.3e12,      # 61.3 TFLOPS (FP32)
            'mi300x': 163.4e12,    # 163.4 TFLOPS (Matrix ops, FP32)
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
            'efficiency_ratio': efficiency_ratio,
            'achieved_tflops': achieved_flops_per_sec / 1e12,
            'peak_tflops': device_peak_flops / 1e12
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
        
        # Evoformer has significant intermediate activations
        batch_size = flops_data['profiling_config']['batch_size']
        seq_len = flops_data['profiling_config']['sequence_length']
        msa_seqs = flops_data['profiling_config']['msa_sequences']
        config = model_info['config']
        
        # MSA activations: B * N * L * d_msa
        msa_activation_size = batch_size * msa_seqs * seq_len * config['msa_dim'] * 4
        # Pair activations: B * L * L * d_pair
        pair_activation_size = batch_size * seq_len * seq_len * config['pair_dim'] * 4
        
        activation_size_estimate = msa_activation_size + pair_activation_size

        # Memory transfers per step (rough estimate)
        # Parameters read once, activations multiple times (forward + 2x backward estimate)
        memory_bytes_per_step = param_size_bytes + (activation_size_estimate * 3)

        avg_time = perf_metrics.get('avg_time_per_step', 1.0)
        memory_bandwidth_used = memory_bytes_per_step / avg_time if avg_time > 0 else 0

        # Arithmetic intensity (FLOPS per byte)
        avg_flops = flops_data['flops_analysis']['avg_flops_per_step']
        arithmetic_intensity = avg_flops / memory_bytes_per_step if memory_bytes_per_step > 0 else 0

        # Get device memory bandwidth
        device_memory_bandwidth = self._get_device_memory_bandwidth()

        intensity_analysis = {
            'arithmetic_intensity_flops_per_byte': arithmetic_intensity,
            'memory_bandwidth_used_gb_per_sec': memory_bandwidth_used / 1e9,
            'memory_bandwidth_utilization_percent': (memory_bandwidth_used / device_memory_bandwidth) * 100 if device_memory_bandwidth > 0 else 0,
            'device_memory_bandwidth_gb_per_sec': device_memory_bandwidth / 1e9,
            'memory_bound_vs_compute_bound': 'memory_bound' if arithmetic_intensity < 10 else 'compute_bound',
            'memory_breakdown': {
                'parameters_mb': param_size_bytes / 1e6,
                'msa_activations_mb': msa_activation_size / 1e6,
                'pair_activations_mb': pair_activation_size / 1e6,
                'total_memory_per_step_mb': memory_bytes_per_step / 1e6
            },
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
        print(f"   Memory Bandwidth Utilization: {intensity_analysis['memory_bandwidth_utilization_percent']:.1f}%")
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
            'mi300x': 5300e9,     # 5300 GB/s (HBM3)
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
            'model_name': 'Tiny OpenFold V1 Baseline - Evoformer',
            'timestamp': datetime.now().isoformat(),
            'device_info': {
                'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'peak_flops': flops_data['efficiency_metrics']['device_peak_flops'],
                'peak_tflops': flops_data['efficiency_metrics']['peak_tflops'],
                'peak_memory_bandwidth': intensity_data.get('device_memory_bandwidth_gb_per_sec', 0) * 1e9
            },
            'performance_point': {
                'arithmetic_intensity': intensity_data.get('arithmetic_intensity_flops_per_byte', 0),
                'achieved_performance': flops_data['performance_metrics']['flops_per_sec'],
                'achieved_tflops': flops_data['efficiency_metrics']['achieved_tflops'],
                'mfu_percent': flops_data['efficiency_metrics']['mfu_percent']
            },
            'evoformer_breakdown': flops_data['flops_analysis']['evoformer_breakdown'],
            'optimization_targets': self._generate_optimization_targets(flops_data, intensity_data)
        }

        # Save roofline data
        roofline_path = output_path / "roofline_data.json"
        with open(roofline_path, 'w') as f:
            json.dump(roofline_data, f, indent=2)

        print(f"Roofline data generated: {roofline_path}")
        return str(roofline_path)

    def _generate_optimization_targets(self, flops_data: Dict, intensity_data: Dict) -> List[Dict[str, str]]:
        """Generate optimization targets based on Evoformer analysis."""
        targets = []

        # MFU-based recommendations
        mfu = flops_data['efficiency_metrics']['mfu_percent']
        if mfu < 30:
            targets.append({
                'target': 'Kernel Fusion - Evoformer Operations',
                'reason': f'Low MFU ({mfu:.1f}%) indicates kernel launch overhead',
                'expected_improvement': '2-3x speedup potential with fused attention and triangle ops'
            })

        # Arithmetic intensity recommendations
        ai = intensity_data.get('arithmetic_intensity_flops_per_byte', 0)
        if ai < 10:
            targets.append({
                'target': 'Memory Optimization',
                'reason': f'Low arithmetic intensity ({ai:.2f}) indicates memory-bound operations',
                'expected_improvement': 'Flash Attention for MSA, gradient checkpointing, activation recomputation'
            })

        # Evoformer-specific optimizations
        breakdown = flops_data['flops_analysis']['evoformer_breakdown']
        
        # Triangle multiplication optimization
        triangle_flops = breakdown.get('triangle_multiplication', 0)
        total_flops = sum(breakdown.values())
        if triangle_flops / total_flops > 0.2:
            targets.append({
                'target': 'Triangle Multiplication Fusion',
                'reason': f'Triangle mult uses {triangle_flops/total_flops*100:.1f}% of FLOPS',
                'expected_improvement': '30-40% reduction with custom fused kernels'
            })

        # MSA attention optimization
        msa_attn_flops = breakdown.get('msa_attention', 0)
        if msa_attn_flops / total_flops > 0.15:
            targets.append({
                'target': 'MSA Attention Optimization',
                'reason': f'MSA attention uses {msa_attn_flops/total_flops*100:.1f}% of FLOPS',
                'expected_improvement': 'Flash Attention adaptation for MSA: 2-3x speedup possible'
            })

        # Outer product mean optimization
        targets.append({
            'target': 'Outer Product Mean Fusion',
            'reason': 'Creates large intermediate pair representation',
            'expected_improvement': '20-30% reduction with memory-efficient implementation'
        })

        # General recommendations
        targets.extend([
            {
                'target': 'Mixed Precision Training (FP16/BF16)',
                'reason': 'Evoformer has many matmul operations suitable for tensor cores',
                'expected_improvement': '2-3x speedup on modern GPUs with tensor cores'
            },
            {
                'target': 'Gradient Checkpointing',
                'reason': 'Large MSA and pair representations consume significant memory',
                'expected_improvement': '3-4x memory reduction, ~20% compute overhead'
            }
        ])

        return targets


def main():
    """Main entry point for DeepSpeed FLOPS analysis."""
    parser = argparse.ArgumentParser(description='DeepSpeed FLOPS Profiler for Tiny OpenFold V1')

    # Model configuration
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for profiling')
    parser.add_argument('--seq-len', type=int, default=64, help='Sequence length')
    parser.add_argument('--num-seqs', type=int, default=16, help='Number of MSA sequences')
    parser.add_argument('--msa-dim', type=int, default=64, help='MSA dimension')
    parser.add_argument('--pair-dim', type=int, default=128, help='Pair dimension')
    parser.add_argument('--num-blocks', type=int, default=4, help='Number of Evoformer blocks')

    # Profiling configuration
    parser.add_argument('--num-steps', type=int, default=10, help='Number of profiling steps')
    parser.add_argument('--output-dir', type=str, default='./flops_analysis', help='Output directory')
    parser.add_argument('--detailed-analysis', action='store_true', help='Enable detailed FLOPS breakdown')
    
    # Device configuration
    parser.add_argument('--device', type=int, default=None, help='Specific GPU device ID to use (e.g., 0, 1, 2)')
    parser.add_argument('--multi-gpu', action='store_true', help='Profile across all available GPUs')
    parser.add_argument('--devices', type=str, default=None, help='Comma-separated list of GPU IDs (e.g., "0,1,2")')

    # Analysis options
    parser.add_argument('--analyze-results', type=str, help='Analyze existing FLOPS results file')
    parser.add_argument('--generate-roofline', action='store_true', help='Generate roofline analysis data')
    parser.add_argument('--computational-intensity', action='store_true', help='Analyze computational intensity')

    args = parser.parse_args()

    if not DEEPSPEED_AVAILABLE and not args.analyze_results:
        print("=" * 70)
        print("DeepSpeed not available. Please install DeepSpeed for FLOPS profiling.")
        print("   pip install deepspeed")
        print("\nAlternatively, this script can still provide FLOPS estimates without DeepSpeed.")
        print("=" * 70)
        return

    # Create analyzer
    analyzer = EvoformerFLOPSAnalyzer(args.output_dir)

    print("=" * 70)
    print("DEEPSPEED FLOPS PROFILER - TINY OPENFOLD V1 (EVOFORMER)")
    print("=" * 70)

    try:
        # Analyze existing results
        if args.analyze_results:
            with open(args.analyze_results, 'r') as f:
                flops_data = json.load(f)
            print(f"📁 Analyzing existing results: {args.analyze_results}")
            
            # Print summary
            print(f"\nModel: {flops_data['model_info']['architecture']}")
            print(f"Parameters: {flops_data['model_info']['total_params']:,}")
            print(f"FLOPS per step: {flops_data['flops_analysis']['avg_flops_per_step']:.2e}")
            print(f"MFU: {flops_data['efficiency_metrics']['mfu_percent']:.1f}%")
            
            return

        # Run new FLOPS profiling
        config = TinyOpenFoldConfig(
            msa_dim=args.msa_dim,
            pair_dim=args.pair_dim,
            n_evoformer_blocks=args.num_blocks,
            n_seqs=args.num_seqs,
            max_seq_len=args.seq_len
        )

        # Determine profiling mode: single GPU vs multi-GPU
        if args.multi_gpu or args.devices:
            # Multi-GPU profiling
            device_ids = None
            if args.devices:
                # Parse comma-separated device IDs
                device_ids = [int(d.strip()) for d in args.devices.split(',')]
            
            flops_results = analyzer.profile_multi_gpu_flops(
                config=config,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                device_ids=device_ids
            )
        else:
            # Single GPU profiling
            flops_results = analyzer.profile_model_flops(
                config=config,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                detailed_analysis=args.detailed_analysis,
                device_id=args.device
            )

        if 'error' in flops_results:
            print(f"⚠️  FLOPS profiling failed: {flops_results['error']}")
            return

        # Computational intensity analysis (only for single GPU)
        if args.computational_intensity and not (args.multi_gpu or args.devices):
            intensity_results = analyzer.analyze_computational_intensity(flops_results)
            if 'error' not in intensity_results:
                print("✓ Computational intensity analysis completed")

        # Generate roofline data (only for single GPU)
        if args.generate_roofline and not (args.multi_gpu or args.devices):
            roofline_path = analyzer.generate_roofline_data(args.output_dir)
            print(f"✓ Roofline data generated: {roofline_path}")

        print(f"\n{'='*70}")
        print(f"FLOPS ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"📁 Results saved to: {args.output_dir}")
        
        # Print metrics based on profiling mode
        if args.multi_gpu or args.devices:
            # Multi-GPU metrics
            print(f"\nMulti-GPU Key Metrics:")
            print(f"   Number of GPUs: {flops_results['multi_gpu_config']['num_gpus']}")
            print(f"   Average MFU: {flops_results['aggregate_metrics']['avg_mfu_percent']:.1f}%")
            print(f"   MFU Std Dev: {flops_results['aggregate_metrics']['mfu_std_dev']:.1f}%")
            print(f"   Total System TFLOPS: {flops_results['aggregate_metrics']['total_system_tflops']:.2f}")
            print(f"   Avg TFLOPS per GPU: {flops_results['aggregate_metrics']['avg_tflops_per_gpu']:.2f}")
            print(f"   Total Throughput: {flops_results['aggregate_metrics']['total_throughput']:.1f} samples/sec")
            print(f"   Multi-GPU Efficiency: {flops_results['aggregate_metrics']['multi_gpu_efficiency_percent']:.1f}%")
            print(f"   Speedup vs Single GPU: {flops_results['comparison']['speedup']:.2f}x")
        else:
            # Single GPU metrics
            print(f"\nSingle GPU Key Metrics:")
            print(f"   Model FLOPS Utilization (MFU): {flops_results['efficiency_metrics']['mfu_percent']:.1f}%")
            print(f"   Achieved TFLOPS: {flops_results['efficiency_metrics']['achieved_tflops']:.2f}")
            print(f"   Peak TFLOPS: {flops_results['efficiency_metrics']['peak_tflops']:.2f}")
            print(f"   Throughput: {flops_results['performance_metrics']['throughput_samples_per_sec']:.1f} samples/sec")
            print(f"   FLOPS per parameter: {flops_results['flops_analysis']['flops_per_parameter']:.2f}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

