#!/usr/bin/env python3
"""
PyTorch Profiler Integration for Tiny OpenFold V2 (Fused)

This script provides enhanced PyTorch profiler integration with fusion-specific analysis,
kernel reduction tracking, and comprehensive performance characterization.

Features:
- Fusion-specific profiling and analysis
- Kernel count reduction measurement
- Flash Attention performance tracking
- Memory bandwidth utilization analysis
- Comparison with baseline (V1)
- Chrome trace export for detailed timeline analysis
- Operator-level performance breakdown with fusion impact
- Bottleneck identification for fused operations
- TensorBoard integration for visualization

Usage:
    # Run profiling with default settings (all fusions enabled)
    python run_pytorch_profiler.py

    # Custom profiling configuration
    python run_pytorch_profiler.py --batch-size 8 --profile-steps 10

    # Ablation study: disable specific fusions
    python run_pytorch_profiler.py --disable-flash-attention

    # Compare with V1 baseline
    python run_pytorch_profiler.py --compare-with-v1 ../version1_pytorch_baseline/pytorch_profiles

    # Generate detailed report
    python run_pytorch_profiler.py --generate-report --output-dir ./analysis
"""

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
import json
import os
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import the model from tiny_openfold_v2
from tiny_openfold_v2 import (
    TinyOpenFoldV2, TinyOpenFoldConfig, FusionConfig, ProteinDataset,
    setup_deterministic_environment, FLASH_ATTENTION_AVAILABLE, TORCH_COMPILE_AVAILABLE
)


def get_gpu_time_total(event) -> float:
    """
    Get GPU time total in a ROCm-compatible way.
    
    On ROCm, PyTorch may expose 'device_time_total' instead of 'cuda_time_total'.
    This function checks for both attributes to ensure compatibility.
    
    Args:
        event: FunctionEventAvg object from PyTorch profiler
        
    Returns:
        GPU time in microseconds (0 if not available)
    """
    if hasattr(event, 'device_time_total'):
        return event.device_time_total
    return getattr(event, 'cuda_time_total', 0)


class FusedProfilerAnalyzer:
    """Advanced PyTorch profiler analysis for fused Evoformer implementation."""

    def __init__(self, profile_dir: str):
        self.profile_dir = Path(profile_dir)
        self.profile_data = None
        self.analysis_results = {}
        self.fusion_stats = {}
        self.throughput_stats = {}

    def run_profiling(
        self,
        config: TinyOpenFoldConfig,
        fusion_config: FusionConfig,
        batch_size: int = 4,
        num_steps: int = 20,
        warmup_steps: int = 3,
        profile_steps: int = 5,
        include_memory: bool = True,
        include_shapes: bool = True,
        device_id: Optional[int] = None
    ) -> profile:
        """Run comprehensive PyTorch profiling session with fusion analysis."""

        print(f"Starting PyTorch Profiler Analysis - Fused Evoformer Architecture")
        print(f"   Profile directory: {self.profile_dir}")
        print(f"   Batch size: {batch_size}")
        print(f"   Sequence length: {config.max_seq_len}")
        print(f"   MSA sequences: {config.n_seqs}")
        print(f"   Total steps: {num_steps}")
        print(f"   Profile steps: {profile_steps}")
        print(f"   Memory profiling: {include_memory}")

        # Fusion configuration summary
        print(f"\n   Fusion Configuration:")
        print(f"      MSA QKV Fusion: {fusion_config.enable_qkv_fusion_msa}")
        print(f"      Triangle QKV Fusion: {fusion_config.enable_qkv_fusion_triangle}")
        print(f"      Flash Attention: {fusion_config.enable_flash_attention and FLASH_ATTENTION_AVAILABLE}")
        print(f"      Triangle Fusion: {fusion_config.enable_triangle_fusion}")
        print(f"      Torch Compile: {fusion_config.enable_torch_compile and TORCH_COMPILE_AVAILABLE}")

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
            print(f"   Using device: {device}")

        # Create model and dataset
        model = TinyOpenFoldV2(config, fusion_config).to(device)
        
        # Apply torch.compile if enabled
        if fusion_config.enable_torch_compile and TORCH_COMPILE_AVAILABLE:
            print("   Applying torch.compile...")
            model = torch.compile(model, mode=fusion_config.torch_compile_mode)

        # Get fusion statistics
        if hasattr(model, 'get_fusion_statistics'):
            self.fusion_stats = model.get_fusion_statistics()
        elif hasattr(model, '_orig_mod'):
            self.fusion_stats = model._orig_mod.get_fusion_statistics()

        dataset = ProteinDataset(config)
        optimizer = torch.optim.AdamW(
            model.parameters() if isinstance(model, nn.Module) else model._orig_mod.parameters(),
            lr=3e-4
        )

        # Ensure profile directory exists
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        # Configure profiler
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        prof = profile(
            activities=activities,
            record_shapes=include_shapes,
            profile_memory=include_memory,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
            schedule=torch.profiler.schedule(
                wait=warmup_steps,
                warmup=1,
                active=profile_steps,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.profile_dir))
        )

        # Training loop with profiling
        model.train()
        
        # Warmup without profiling
        print("\n   Running warmup steps...")
        for step in range(warmup_steps):
            msa_tokens, pair_features, target_distances = dataset.get_batch(batch_size)
            msa_tokens = msa_tokens.to(device)
            pair_features = pair_features.to(device)
            target_distances = target_distances.to(device)

            outputs = model(msa_tokens, pair_features, target_distances)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Profiled steps with timing
        print(f"   Running {num_steps} steps with profiling...")
        prof.start()
        
        # Track timing for throughput calculation
        step_times = []
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        
        for step in range(num_steps):
            step_start = time.time()
            
            msa_tokens, pair_features, target_distances = dataset.get_batch(batch_size)
            msa_tokens = msa_tokens.to(device)
            pair_features = pair_features.to(device)
            target_distances = target_distances.to(device)

            outputs = model(msa_tokens, pair_features, target_distances)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            prof.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_time = time.time() - step_start
            step_times.append(step_time)

            if step % 5 == 0:
                print(f"      Step {step}/{num_steps} - Loss: {loss.item():.4f}")

        prof.stop()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        # Calculate throughput statistics
        total_samples = num_steps * batch_size
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        avg_throughput = batch_size / avg_step_time if avg_step_time > 0 else 0
        
        self.throughput_stats = {
            'total_steps': num_steps,
            'batch_size': batch_size,
            'total_samples': total_samples,
            'total_time_sec': total_time,
            'avg_step_time_ms': avg_step_time * 1000,
            'avg_throughput_samples_per_sec': avg_throughput,
            'min_step_time_ms': min(step_times) * 1000 if step_times else 0,
            'max_step_time_ms': max(step_times) * 1000 if step_times else 0
        }

        self.profile_data = prof
        print("\n   Profiling complete!")
        
        return prof

    def analyze_fusion_impact(self) -> Dict[str, Any]:
        """Analyze the impact of fusion optimizations."""
        if self.profile_data is None:
            return {"error": "No profiling data available"}

        print("\nAnalyzing fusion impact...")
        
        # Get operator statistics
        events = self.profile_data.key_averages()
        
        # Categorize operators by fusion type
        fusion_categories = {
            'fused_qkv': [],
            'flash_attention': [],
            'fused_triangle': [],
            'standard_ops': []
        }

        for event in events:
            name = event.key
            if 'fused_qkv' in name or 'qkv_fused' in name:
                fusion_categories['fused_qkv'].append(event)
            elif 'flash_attention' in name:
                fusion_categories['flash_attention'].append(event)
            elif 'fused_triangle' in name or 'triangle.*fused' in name:
                fusion_categories['fused_triangle'].append(event)
            else:
                fusion_categories['standard_ops'].append(event)

        # Calculate fusion statistics
        fusion_analysis = {}
        for category, events_list in fusion_categories.items():
            if events_list:
                total_time = sum(get_gpu_time_total(e) if torch.cuda.is_available() else e.cpu_time_total 
                               for e in events_list)
                total_calls = sum(e.count for e in events_list)
                fusion_analysis[category] = {
                    'total_time_ms': total_time / 1000.0,
                    'total_calls': total_calls,
                    'avg_time_per_call_ms': (total_time / total_calls / 1000.0) if total_calls > 0 else 0
                }

        self.analysis_results['fusion_impact'] = fusion_analysis
        return fusion_analysis

    def analyze_memory_efficiency(self) -> Dict[str, Any]:
        """Analyze memory efficiency improvements from fusion."""
        if self.profile_data is None:
            return {"error": "No profiling data available"}

        print("Analyzing memory efficiency...")
        
        events = self.profile_data.key_averages()
        
        # Track memory-intensive operations
        memory_analysis = {
            'attention_memory': 0,
            'triangle_memory': 0,
            'total_memory': 0,
            'peak_memory_mb': 0
        }

        if torch.cuda.is_available():
            memory_analysis['peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024**2)

        for event in events:
            if hasattr(event, 'cpu_memory_usage') and event.cpu_memory_usage > 0:
                memory_usage = event.cpu_memory_usage / (1024**2)  # Convert to MB
                memory_analysis['total_memory'] += memory_usage
                
                if 'attention' in event.key:
                    memory_analysis['attention_memory'] += memory_usage
                elif 'triangle' in event.key:
                    memory_analysis['triangle_memory'] += memory_usage

        self.analysis_results['memory_efficiency'] = memory_analysis
        return memory_analysis

    def generate_comprehensive_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive profiling report with fusion analysis."""
        
        if output_file is None:
            output_file = self.profile_dir / "comprehensive_profiling_report.md"
        
        report_lines = []
        report_lines.append("# Tiny OpenFold V2 - Fused Implementation Profiling Report")
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Configuration summary
        report_lines.append("## Configuration")
        report_lines.append("\n### Fusion Settings")
        if self.fusion_stats:
            report_lines.append(f"- MSA QKV Fusion: {'Enabled' if self.fusion_stats.get('qkv_fusion_msa_enabled') else 'Disabled'}")
            report_lines.append(f"- Triangle QKV Fusion: {'Enabled' if self.fusion_stats.get('qkv_fusion_triangle_enabled') else 'Disabled'}")
            report_lines.append(f"- Flash Attention: {'Enabled' if self.fusion_stats.get('flash_attention_enabled') else 'Disabled'}")
            report_lines.append(f"- Triangle Fusion: {'Enabled' if self.fusion_stats.get('triangle_fusion_enabled') else 'Disabled'}")
            report_lines.append(f"- Torch Compile: {'Enabled' if self.fusion_stats.get('torch_compile_enabled') else 'Disabled'}")
            report_lines.append(f"\n### Kernel Reduction")
            report_lines.append(f"- Baseline kernels per block: {self.fusion_stats.get('baseline_kernels_per_block', 'N/A')}")
            report_lines.append(f"- Fused kernels per block: {self.fusion_stats.get('fused_kernels_per_block', 'N/A')}")
            report_lines.append(f"- Kernel reduction: {self.fusion_stats.get('kernel_reduction_percent', 0):.1f}%")
            report_lines.append(f"- Total kernels saved: {self.fusion_stats.get('total_kernel_reduction', 'N/A')}")
        
        # Performance analysis
        if self.profile_data:
            report_lines.append("\n## Performance Analysis")
            
            events = self.profile_data.key_averages()
            
            # Top operations by time
            report_lines.append("\n### Top 15 Operations by GPU Time")
            report_lines.append("\n| Operation | GPU Time (ms) | CPU Time (ms) | Calls | Avg Time (ms) |")
            report_lines.append("|-----------|---------------|---------------|-------|---------------|")
            
            sorted_events = sorted(events, 
                                 key=lambda e: get_gpu_time_total(e) if torch.cuda.is_available() else e.cpu_time_total,
                                 reverse=True)[:15]
            
            for event in sorted_events:
                gpu_time = get_gpu_time_total(event) / 1000.0 if torch.cuda.is_available() else 0
                cpu_time = event.cpu_time_total / 1000.0
                avg_time = gpu_time / event.count if event.count > 0 else 0
                report_lines.append(f"| {event.key[:50]} | {gpu_time:.2f} | {cpu_time:.2f} | {event.count} | {avg_time:.3f} |")
        
        # Fusion impact analysis
        if 'fusion_impact' in self.analysis_results:
            report_lines.append("\n### Fusion Impact Analysis")
            fusion_impact = self.analysis_results['fusion_impact']
            
            for category, stats in fusion_impact.items():
                if stats['total_calls'] > 0:
                    report_lines.append(f"\n**{category}:**")
                    report_lines.append(f"- Total time: {stats['total_time_ms']:.2f} ms")
                    report_lines.append(f"- Total calls: {stats['total_calls']}")
                    report_lines.append(f"- Average time per call: {stats['avg_time_per_call_ms']:.3f} ms")
        
        # Memory analysis
        if 'memory_efficiency' in self.analysis_results:
            report_lines.append("\n### Memory Efficiency")
            mem_analysis = self.analysis_results['memory_efficiency']
            
            report_lines.append(f"- Peak memory: {mem_analysis['peak_memory_mb']:.1f} MB")
            report_lines.append(f"- Attention memory: {mem_analysis['attention_memory']:.1f} MB")
            report_lines.append(f"- Triangle memory: {mem_analysis['triangle_memory']:.1f} MB")
            report_lines.append(f"- Total tracked memory: {mem_analysis['total_memory']:.1f} MB")
        
        # Recommendations
        report_lines.append("\n## Optimization Recommendations")
        report_lines.append("\n### Based on Profiling Results:")
        
        if self.fusion_stats.get('flash_attention_enabled'):
            report_lines.append("- ✓ Flash Attention is enabled - memory efficiency optimized")
        else:
            report_lines.append("- ⚠ Consider enabling Flash Attention for memory savings")
        
        if self.fusion_stats.get('qkv_fusion_msa_enabled'):
            report_lines.append("- ✓ MSA QKV fusion is enabled - kernel launch overhead reduced")
        else:
            report_lines.append("- ⚠ Enable MSA QKV fusion to reduce kernel launches")
        
        if self.fusion_stats.get('triangle_fusion_enabled'):
            report_lines.append("- ✓ Triangle fusion is enabled - triangle operations optimized")
        else:
            report_lines.append("- ⚠ Enable triangle fusion for better performance")
        
        # Write report
        report_content = "\n".join(report_lines)
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        print(f"\nComprehensive report saved to: {output_file}")
        return report_content

    def get_throughput_summary(self) -> Dict[str, Any]:
        """Get throughput summary statistics."""
        return self.throughput_stats
    
    def export_analysis(self, output_file: Optional[str] = None):
        """Export analysis results to JSON."""
        if output_file is None:
            output_file = self.profile_dir / "fusion_analysis.json"
        
        export_data = {
            'fusion_statistics': self.fusion_stats,
            'analysis_results': self.analysis_results,
            'throughput_statistics': self.throughput_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Analysis exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='PyTorch Profiler for Tiny OpenFold V2 (Fused)')
    
    # Model configuration
    parser.add_argument('--msa-dim', type=int, default=64, help='MSA dimension')
    parser.add_argument('--pair-dim', type=int, default=128, help='Pair dimension')
    parser.add_argument('--num-blocks', type=int, default=4, help='Number of Evoformer blocks')
    parser.add_argument('--num-seqs', type=int, default=16, help='Number of MSA sequences')
    parser.add_argument('--seq-len', type=int, default=64, help='Sequence length')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-steps', type=int, default=20, help='Total steps including warmup')
    parser.add_argument('--warmup-steps', type=int, default=3, help='Warmup steps')
    parser.add_argument('--profile-steps', type=int, default=5, help='Steps to profile')
    parser.add_argument('--device', type=int, default=None, help='GPU device ID')
    
    # Fusion configuration
    parser.add_argument('--disable-qkv-fusion-msa', action='store_true', help='Disable MSA QKV fusion')
    parser.add_argument('--disable-qkv-fusion-triangle', action='store_true', help='Disable triangle QKV fusion')
    parser.add_argument('--disable-flash-attention', action='store_true', help='Disable Flash Attention')
    parser.add_argument('--disable-triangle-fusion', action='store_true', help='Disable triangle fusion')
    parser.add_argument('--enable-torch-compile', action='store_true', help='Enable torch.compile')
    parser.add_argument('--disable-all-fusion', action='store_true', help='Disable all fusion (baseline mode)')
    
    # Profiling configuration
    parser.add_argument('--profile-dir', type=str, default='./pytorch_profiles_v2', help='Profile output directory')
    parser.add_argument('--no-memory', action='store_true', help='Disable memory profiling')
    parser.add_argument('--no-shapes', action='store_true', help='Disable shape recording')
    parser.add_argument('--generate-report', action='store_true', default=True, help='Generate comprehensive report')
    parser.add_argument('--compare-with-v1', type=str, help='Path to V1 profiling results for comparison')
    
    args = parser.parse_args()
    
    # Configure model
    config = TinyOpenFoldConfig(
        msa_dim=args.msa_dim,
        pair_dim=args.pair_dim,
        n_evoformer_blocks=args.num_blocks,
        n_seqs=args.num_seqs,
        max_seq_len=args.seq_len,
        msa_intermediate_dim=args.msa_dim * 4,
        pair_intermediate_dim=args.pair_dim * 4
    )
    
    # Configure fusion
    if args.disable_all_fusion:
        fusion_config = FusionConfig(
            enable_qkv_fusion_msa=False,
            enable_qkv_fusion_triangle=False,
            enable_flash_attention=False,
            enable_triangle_fusion=False,
            enable_torch_compile=False
        )
    else:
        fusion_config = FusionConfig(
            enable_qkv_fusion_msa=not args.disable_qkv_fusion_msa,
            enable_qkv_fusion_triangle=not args.disable_qkv_fusion_triangle,
            enable_flash_attention=not args.disable_flash_attention,
            enable_triangle_fusion=not args.disable_triangle_fusion,
            enable_torch_compile=args.enable_torch_compile
        )
    
    # Create analyzer and run profiling
    analyzer = FusedProfilerAnalyzer(args.profile_dir)
    
    try:
        prof = analyzer.run_profiling(
            config=config,
            fusion_config=fusion_config,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
            include_memory=not args.no_memory,
            include_shapes=not args.no_shapes,
            device_id=args.device
        )
        
        # Analyze results
        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)
        
        fusion_impact = analyzer.analyze_fusion_impact()
        memory_efficiency = analyzer.analyze_memory_efficiency()
        
        # Generate report
        if args.generate_report:
            analyzer.generate_comprehensive_report()
        
        # Export analysis
        analyzer.export_analysis()
        
        # Print throughput summary
        throughput_stats = analyzer.get_throughput_summary()
        if throughput_stats:
            print("\n" + "="*70)
            print("THROUGHPUT SUMMARY")
            print("="*70)
            print(f"   Total steps:           {throughput_stats['total_steps']}")
            print(f"   Batch size:            {throughput_stats['batch_size']}")
            print(f"   Total samples:         {throughput_stats['total_samples']}")
            print(f"   Total time:            {throughput_stats['total_time_sec']:.2f} seconds")
            print(f"   Average step time:     {throughput_stats['avg_step_time_ms']:.2f} ms")
            print(f"   Average throughput:     {throughput_stats['avg_throughput_samples_per_sec']:.2f} samples/sec")
            print(f"   Min step time:         {throughput_stats['min_step_time_ms']:.2f} ms")
            print(f"   Max step time:         {throughput_stats['max_step_time_ms']:.2f} ms")
            print("="*70)
        
        # Print summary
        print("\n" + "="*70)
        print("PROFILING SUMMARY")
        print("="*70)
        print(f"\nProfile directory: {args.profile_dir}")
        print(f"Trace files: {args.profile_dir}/*.pt.trace.json")
        print(f"\nTo visualize:")
        print(f"  1. Chrome trace: Open chrome://tracing and load trace file")
        print(f"  2. TensorBoard: tensorboard --logdir {args.profile_dir}")
        print(f"\nReports generated:")
        print(f"  - comprehensive_profiling_report.md")
        print(f"  - fusion_analysis.json")
        
        if args.compare_with_v1:
            print(f"\nComparison with V1: {args.compare_with_v1}")
            print("  (Comparison analysis not yet implemented)")
        
    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


