#!/usr/bin/env python3
"""
PyTorch Profiler Integration for Tiny OpenFold V1

This script provides enhanced PyTorch profiler integration with detailed analysis,
visualization, and bottleneck identification capabilities for the Evoformer baseline model.

Features:
- Comprehensive profiler configuration
- Chrome trace export for detailed timeline analysis
- Operator-level performance breakdown
- Memory usage analysis
- Bottleneck identification and recommendations
- TensorBoard integration for visualization
- Evoformer-specific optimization analysis

Usage:
    # Run profiling with default settings
    python run_pytorch_profiler.py

    # Custom profiling configuration
    python run_pytorch_profiler.py --batch-size 8 --profile-steps 10

    # Analyze existing profiling results
    python run_pytorch_profiler.py --analyze-existing ./pytorch_profiles

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
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import the model from tiny_openfold_v1
from tiny_openfold_v1 import TinyOpenFold, TinyOpenFoldConfig, ProteinDataset, setup_deterministic_environment


class PyTorchProfilerAnalyzer:
    """Advanced PyTorch profiler analysis and visualization for Evoformer."""

    def __init__(self, profile_dir: str):
        self.profile_dir = Path(profile_dir)
        self.profile_data = None
        self.analysis_results = {}

    def run_profiling(
        self,
        config: TinyOpenFoldConfig,
        batch_size: int = 4,
        num_steps: int = 20,
        warmup_steps: int = 3,
        profile_steps: int = 5,
        include_memory: bool = True,
        include_shapes: bool = True,
        device_id: Optional[int] = None
    ) -> profile:
        """Run comprehensive PyTorch profiling session."""

        print(f"Starting PyTorch Profiler Analysis - Evoformer Architecture")
        print(f"   Profile directory: {self.profile_dir}")
        print(f"   Batch size: {batch_size}")
        print(f"   Sequence length: {config.max_seq_len}")
        print(f"   MSA sequences: {config.n_seqs}")
        print(f"   Total steps: {num_steps}")
        print(f"   Profile steps: {profile_steps}")
        print(f"   Memory profiling: {include_memory}")

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
        model = TinyOpenFold(config).to(device)
        dataset = ProteinDataset(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Ensure profile directory exists
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        # Configure profiler
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        def trace_handler(prof):
            """Custom trace handler for comprehensive output."""
            # Export Chrome trace for both TensorBoard and direct viewing
            chrome_trace_path = self.profile_dir / f"trace_step_{prof.step_num}.json"
            prof.export_chrome_trace(str(chrome_trace_path))
            
            # Export stacks (if available)
            if hasattr(prof, 'export_stacks'):
                stacks_path = self.profile_dir / f"stacks_step_{prof.step_num}.txt"
                try:
                    prof.export_stacks(str(stacks_path), "self_cpu_time_total")
                except Exception as e:
                    print(f"   Warning: Could not export stacks: {e}")

            print(f"   Exported trace for step {prof.step_num}")

        # Run profiling session
        with profile(
            activities=activities,
            record_shapes=include_shapes,
            profile_memory=include_memory,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            schedule=torch.profiler.schedule(
                wait=warmup_steps,
                warmup=1,
                active=profile_steps,
                repeat=1
            ),
            on_trace_ready=trace_handler
        ) as prof:
            model.train()
            
            # Track timing for throughput
            import time
            step_times = []
            start_time = time.time()

            for step in range(num_steps):
                step_start = time.time()
                
                # Get batch
                msa_tokens, pair_tokens, targets = dataset.get_batch(batch_size)
                msa_tokens = msa_tokens.to(device)
                pair_tokens = pair_tokens.to(device)
                targets = targets.to(device)

                # Forward pass
                with record_function("forward_pass"):
                    outputs = model(msa_tokens, pair_tokens, targets)
                    loss = outputs['loss']

                # Backward pass
                with record_function("backward_pass"):
                    loss.backward()

                # Optimizer step
                with record_function("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad()

                # Profiler step
                prof.step()
                
                # Track step time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_end = time.time()
                step_times.append(step_end - step_start)

                if step % 10 == 0:
                    print(f"   Step {step}/{num_steps}, Loss: {loss.item():.4f}")

            # Calculate and print throughput summary
            total_time = time.time() - start_time
            total_samples = num_steps * batch_size
            avg_step_time = sum(step_times) / len(step_times)
            avg_throughput = batch_size / avg_step_time
            
            print(f"\n{'='*70}")
            print(f"Profiling Throughput Summary:")
            print(f"{'='*70}")
            print(f"   Total steps:           {num_steps}")
            print(f"   Batch size:            {batch_size}")
            print(f"   Total samples:         {total_samples}")
            print(f"   Total time:            {total_time:.2f} seconds")
            print(f"   Average step time:     {avg_step_time*1000:.2f} ms")
            print(f"   Average throughput:    {avg_throughput:.1f} samples/sec")
            print(f"   Min step time:         {min(step_times)*1000:.2f} ms")
            print(f"   Max step time:         {max(step_times)*1000:.2f} ms")
            print(f"{'='*70}\n")

        # Save profiler data for analysis
        self.profile_data = prof
        return prof

    def analyze_operator_performance(self, prof: profile) -> Dict[str, Any]:
        """Analyze operator-level performance characteristics."""
        print(f"\nAnalyzing operator performance...")

        # Get operator statistics
        cpu_stats = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=50)
        cuda_stats = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50) if torch.cuda.is_available() else None

        # Calculate total time for percentage calculation
        total_cpu_time = sum(event.cpu_time_total for event in prof.key_averages())
        total_cuda_time = sum(getattr(event, 'cuda_time_total', 0) for event in prof.key_averages()) if torch.cuda.is_available() else 0

        # Parse operator data
        operator_data = []
        for event in prof.key_averages():
            operator_info = {
                'name': event.key,
                'cpu_time_total': event.cpu_time_total,
                'cpu_time_avg': event.cpu_time / max(1, event.count),
                'cpu_time_percent': (event.cpu_time_total / total_cpu_time * 100) if total_cpu_time > 0 else 0,
                'count': event.count,
                'input_shapes': str(event.input_shapes) if hasattr(event, 'input_shapes') else '',
                'flops': getattr(event, 'flops', 0)
            }

            if torch.cuda.is_available():
                # Avoid accessing deprecated cuda_time attribute
                if hasattr(event, 'device_time'):
                    device_time = event.device_time
                    device_time_total = event.device_time_total
                else:
                    device_time = 0
                    device_time_total = 0

                operator_info.update({
                    'cuda_time_total': device_time_total,
                    'cuda_time_avg': device_time / max(1, event.count),
                    'cuda_memory_usage': getattr(event, 'cuda_memory_usage', 0)
                })

            operator_data.append(operator_info)

        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(operator_data)

        analysis = {
            'operator_stats': operator_data,
            'bottlenecks': bottlenecks,
            'cpu_table': cpu_stats,
            'cuda_table': cuda_stats
        }

        # Save detailed analysis
        analysis_path = self.profile_dir / "operator_analysis.json"
        with open(analysis_path, 'w') as f:
            # Convert non-serializable data
            serializable_data = {
                'operator_stats': operator_data,
                'bottlenecks': bottlenecks,
                'timestamp': datetime.now().isoformat()
            }
            json.dump(serializable_data, f, indent=2)

        return analysis

    def _identify_bottlenecks(self, operator_data: List[Dict]) -> Dict[str, Any]:
        """Identify performance bottlenecks and optimization opportunities for Evoformer."""
        bottlenecks = {
            'top_cpu_time': [],
            'top_cuda_time': [],
            'memory_intensive': [],
            'low_flops_utilization': [],
            'optimization_targets': []
        }

        # Sort by CPU time
        cpu_sorted = sorted(operator_data, key=lambda x: x['cpu_time_total'], reverse=True)
        bottlenecks['top_cpu_time'] = cpu_sorted[:10]

        # Sort by CUDA time (if available)
        if torch.cuda.is_available():
            cuda_sorted = sorted(operator_data, key=lambda x: x.get('cuda_time_total', 0), reverse=True)
            bottlenecks['top_cuda_time'] = cuda_sorted[:10]

            # Memory intensive operations
            memory_sorted = sorted(operator_data, key=lambda x: x.get('cuda_memory_usage', 0), reverse=True)
            bottlenecks['memory_intensive'] = memory_sorted[:10]

        # Identify Evoformer-specific optimization targets
        optimization_targets = []
        for op in operator_data:
            name = op['name'].lower()

            # MSA Attention optimizations
            if any(keyword in name for keyword in ['matmul', 'linear', 'addmm', 'bmm']):
                if 'msa' in name and any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj']):
                    optimization_targets.append({
                        'operation': op['name'],
                        'optimization': 'MSA Attention Fusion',
                        'potential_benefit': 'Fuse MSA Q/K/V projections and implement Flash Attention',
                        'priority': 'high'
                    })

            # Triangle Multiplication optimizations
            if 'triangle' in name and ('multiply' in name or 'einsum' in name):
                optimization_targets.append({
                    'operation': op['name'],
                    'optimization': 'Triangle Multiplication Fusion',
                    'potential_benefit': 'Fuse triangle update operations to reduce kernel launches',
                    'priority': 'high'
                })

            # Outer Product Mean optimizations
            if 'outer_product' in name or ('einsum' in name and 'outer' in name):
                optimization_targets.append({
                    'operation': op['name'],
                    'optimization': 'Outer Product Optimization',
                    'potential_benefit': 'Use optimized einsum implementations or custom kernels',
                    'priority': 'medium'
                })

            # Pair Representation optimizations
            if 'pair' in name and any(keyword in name for keyword in ['linear', 'matmul']):
                optimization_targets.append({
                    'operation': op['name'],
                    'optimization': 'Pair Update Fusion',
                    'potential_benefit': 'Fuse pair update operations',
                    'priority': 'medium'
                })

            # LayerNorm optimizations
            if 'layernorm' in name or 'layer_norm' in name:
                optimization_targets.append({
                    'operation': op['name'],
                    'optimization': 'LayerNorm Fusion',
                    'potential_benefit': 'Fuse LayerNorm with adjacent operations',
                    'priority': 'low'
                })

        bottlenecks['optimization_targets'] = optimization_targets

        return bottlenecks

    def analyze_memory_usage(self, prof: profile) -> Dict[str, Any]:
        """Analyze memory usage patterns and identify optimization opportunities."""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available for memory analysis'}

        print(f"\nAnalyzing memory usage patterns...")

        memory_analysis = {}

        try:
            # Memory timeline analysis
            memory_events = []
            for event in prof.key_averages():
                if hasattr(event, 'cuda_memory_usage') and event.cuda_memory_usage > 0:
                    memory_events.append({
                        'name': event.key,
                        'memory_usage': event.cuda_memory_usage,
                        'count': event.count,
                        'avg_memory_per_call': event.cuda_memory_usage / max(1, event.count)
                    })

            memory_events.sort(key=lambda x: x['memory_usage'], reverse=True)

            memory_analysis = {
                'peak_memory_events': memory_events[:20],
                'total_memory_allocated': sum(event['memory_usage'] for event in memory_events),
                'memory_efficiency_recommendations': self._generate_memory_recommendations(memory_events)
            }

            # Save memory analysis
            memory_path = self.profile_dir / "memory_analysis.json"
            with open(memory_path, 'w') as f:
                json.dump(memory_analysis, f, indent=2)

        except Exception as e:
            memory_analysis = {'error': f'Memory analysis failed: {str(e)}'}

        return memory_analysis

    def _generate_memory_recommendations(self, memory_events: List[Dict]) -> List[str]:
        """Generate memory optimization recommendations for Evoformer."""
        recommendations = []

        # Check for high memory operations
        high_memory_ops = [event for event in memory_events if event['memory_usage'] > 1e6]  # > 1MB

        if high_memory_ops:
            recommendations.append(
                f"High memory operations detected: {len(high_memory_ops)} operations using >1MB. "
                "Consider gradient checkpointing for Evoformer blocks."
            )

        # Check for MSA attention memory patterns
        msa_attention_ops = [event for event in memory_events if 'msa' in event['name'].lower() and 'attention' in event['name'].lower()]
        if msa_attention_ops:
            recommendations.append(
                "MSA attention operations detected. Consider Flash Attention adaptation for memory-efficient MSA computation."
            )

        # Check for triangle operations
        triangle_ops = [event for event in memory_events if 'triangle' in event['name'].lower()]
        if triangle_ops:
            recommendations.append(
                "Triangle operations detected. Memory usage for L²×d pair representations can be reduced with "
                "chunking or gradient checkpointing strategies."
            )

        # Check for temporary tensor creation
        temp_ops = [event for event in memory_events if event['count'] > 100]
        if temp_ops:
            recommendations.append(
                f"High-frequency operations detected: {len(temp_ops)} operations called >100 times. "
                "Consider tensor reuse or pre-allocation strategies, especially for pair representations."
            )

        # Evoformer-specific recommendations
        outer_product_ops = [event for event in memory_events if 'outer_product' in event['name'].lower()]
        if outer_product_ops:
            recommendations.append(
                "Outer product mean operations require O(L²) memory. Consider chunked computation "
                "for longer sequences to reduce peak memory usage."
            )

        return recommendations

    def generate_comprehensive_report(self, output_dir: str = None) -> str:
        """Generate comprehensive profiling report with recommendations."""
        if output_dir is None:
            output_dir = self.profile_dir

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_path = output_path / "comprehensive_profiling_report.md"

        report_content = f"""# PyTorch Profiler Analysis Report - Tiny OpenFold V1 (Evoformer)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Profile Directory:** {self.profile_dir}

## Executive Summary

This report provides comprehensive performance analysis of the Tiny OpenFold V1 baseline implementation
using PyTorch's built-in profiler. The analysis focuses on identifying optimization opportunities
for the Evoformer architecture.

## Evoformer Architecture Overview

The Evoformer consists of several key components:
- **MSA Stack**: Row and column attention over multiple sequence alignments
- **Pair Stack**: Triangle multiplication and attention operations
- **Outer Product Mean**: Combines MSA and pair representations
- **Transitions**: Feed-forward networks for MSA and pair

## Analysis Results

### Top CPU Time Consumers

The following operations consume the most CPU time:

```
{self.analysis_results.get('operator_analysis', {}).get('cpu_table', 'No data available')}
```

### Top CUDA Time Consumers

GPU operations breakdown:

```
{self.analysis_results.get('operator_analysis', {}).get('cuda_table', 'No data available')}
```

### Memory Usage Analysis

{self._format_memory_analysis()}

### Optimization Recommendations

#### High Priority Optimizations (Evoformer-Specific)

{self._format_optimization_recommendations('high')}

#### Medium Priority Optimizations

{self._format_optimization_recommendations('medium')}

## Next Steps for Optimization

Based on this analysis, the following optimizations should be considered:

1. **MSA Attention Optimization**: Adapt Flash Attention for row/column MSA attention
2. **Triangle Operation Fusion**: Fuse triangle multiplication and attention kernels
3. **Memory-Efficient Outer Product**: Implement chunked outer product mean computation
4. **Gradient Checkpointing**: Apply to Evoformer blocks for large sequences
5. **Mixed Precision**: Use FP16/BF16 for improved throughput

## Evoformer-Specific Bottlenecks

### Triangle Operations
- **Complexity**: O(L²) for pair representations
- **Optimization**: Kernel fusion, chunking for long sequences
- **Expected Improvement**: 1.5-2× speedup

### MSA Attention
- **Complexity**: O(N×L) for N sequences of length L
- **Optimization**: Flash Attention adaptation
- **Expected Improvement**: 2-3× speedup, 50% memory reduction

### Outer Product Mean
- **Complexity**: O(N×L²)
- **Optimization**: Chunked computation, low-precision accumulation
- **Expected Improvement**: 1.3-1.5× speedup

## Detailed Analysis Files

- **Operator Analysis**: `operator_analysis.json`
- **Memory Analysis**: `memory_analysis.json`
- **Chrome Traces**: `trace_step_*.json`
- **Performance Summary**: `performance_summary.json`

## Visualization

To visualize the profiling results:

1. **TensorBoard**: `tensorboard --logdir {self.profile_dir}`
2. **Chrome Trace**: Open `trace_step_*.json` in Chrome's chrome://tracing

## Comparison with DeepSpeed FLOPS Profiler

For computational efficiency analysis (MFU, FLOPS breakdown), run:
```bash
./run_deepspeed_flops.sh --device 0 --num-steps 50
```

See `PROFILER_RESULTS_COMPARISON.md` for side-by-side comparison.

---
*This report was generated by the TinyOpenFold profiling tools.*
"""

        with open(report_path, 'w') as f:
            f.write(report_content)

        print(f"Comprehensive report generated: {report_path}")
        return str(report_path)

    def _format_memory_analysis(self) -> str:
        """Format memory analysis for report."""
        memory_data = self.analysis_results.get('memory_analysis', {})

        if 'error' in memory_data:
            return f"Memory analysis unavailable: {memory_data['error']}"

        peak_events = memory_data.get('peak_memory_events', [])[:5]

        if not peak_events:
            return "No memory usage data available."

        formatted = "**Top Memory Consumers:**\n\n"
        for i, event in enumerate(peak_events, 1):
            formatted += f"{i}. {event['name']}: {event['memory_usage']/1e6:.1f} MB\n"

        recommendations = memory_data.get('memory_efficiency_recommendations', [])
        if recommendations:
            formatted += "\n**Memory Optimization Recommendations:**\n\n"
            for rec in recommendations:
                formatted += f"- {rec}\n"

        return formatted

    def _format_optimization_recommendations(self, priority: str) -> str:
        """Format optimization recommendations by priority."""
        bottlenecks = self.analysis_results.get('operator_analysis', {}).get('bottlenecks', {})
        targets = bottlenecks.get('optimization_targets', [])

        priority_targets = [target for target in targets if target.get('priority') == priority]

        if not priority_targets:
            return f"No {priority} priority optimizations identified."

        formatted = ""
        for target in priority_targets:
            formatted += f"- **{target['optimization']}**: {target['potential_benefit']}\n"
            formatted += f"  - Operation: {target['operation']}\n\n"

        return formatted

    def analyze_existing_profiles(self, profile_dir: str):
        """Analyze existing profiling results from a directory."""
        profile_path = Path(profile_dir)

        if not profile_path.exists():
            print(f"Profile directory not found: {profile_dir}")
            return

        # Look for JSON trace files
        trace_files = list(profile_path.glob("trace_step_*.json"))

        if not trace_files:
            print(f"No trace files found in: {profile_dir}")
            return

        print(f"Analyzing existing profiles from: {profile_dir}")
        print(f"   Found {len(trace_files)} trace files")

        # Analyze each trace file
        for trace_file in trace_files:
            print(f"   Analyzing: {trace_file.name}")
            # Note: Full trace analysis would require parsing the Chrome trace format
            # For now, we'll provide summary information

        print("Analysis of existing profiles completed")


def main():
    """Main entry point for PyTorch profiler analysis."""
    parser = argparse.ArgumentParser(description='PyTorch Profiler for Tiny OpenFold V1')

    # Model configuration
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for profiling')
    parser.add_argument('--seq-len', type=int, default=64, help='Sequence length')
    parser.add_argument('--num-seqs', type=int, default=16, help='Number of MSA sequences')
    parser.add_argument('--msa-dim', type=int, default=64, help='MSA dimension')
    parser.add_argument('--pair-dim', type=int, default=128, help='Pair dimension')
    parser.add_argument('--num-blocks', type=int, default=4, help='Number of Evoformer blocks')

    # Profiling configuration
    parser.add_argument('--num-steps', type=int, default=20, help='Total profiling steps')
    parser.add_argument('--warmup-steps', type=int, default=3, help='Warmup steps')
    parser.add_argument('--profile-steps', type=int, default=5, help='Active profiling steps')
    parser.add_argument('--profile-dir', type=str, default='./pytorch_profiles', help='Profile output directory')
    parser.add_argument('--device', type=int, default=None, help='GPU device ID (e.g., 0, 1, 2)')

    # Analysis options
    parser.add_argument('--include-memory', action='store_true', default=True, help='Include memory profiling')
    parser.add_argument('--include-shapes', action='store_true', default=True, help='Include tensor shapes')
    parser.add_argument('--analyze-existing', type=str, help='Analyze existing profile directory')
    parser.add_argument('--generate-report', action='store_true', help='Generate comprehensive report')
    parser.add_argument('--output-dir', type=str, help='Output directory for reports')

    args = parser.parse_args()

    # Create analyzer
    analyzer = PyTorchProfilerAnalyzer(args.profile_dir)

    # Analyze existing profiles
    if args.analyze_existing:
        analyzer.analyze_existing_profiles(args.analyze_existing)
        return

    # Run new profiling session
    config = TinyOpenFoldConfig(
        msa_dim=args.msa_dim,
        pair_dim=args.pair_dim,
        n_evoformer_blocks=args.num_blocks,
        n_seqs=args.num_seqs,
        max_seq_len=args.seq_len
    )

    print("PYTORCH PROFILER - TINY OPENFOLD V1 (EVOFORMER) ANALYSIS")
    print("=" * 70)

    try:
        # Run profiling
        prof = analyzer.run_profiling(
            config=config,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
            include_memory=args.include_memory,
            include_shapes=args.include_shapes,
            device_id=args.device
        )

        # Analyze results
        print("\n" + "="*70)
        analyzer.analysis_results['operator_analysis'] = analyzer.analyze_operator_performance(prof)
        analyzer.analysis_results['memory_analysis'] = analyzer.analyze_memory_usage(prof)

        # Generate report
        if args.generate_report:
            report_path = analyzer.generate_comprehensive_report(args.output_dir)
            print(f"\nReport generated: {report_path}")

        print(f"\nProfiling analysis completed successfully!")
        print(f"Results saved to: {args.profile_dir}")
        print(f"\nNext steps:")
        print(f"   1. Launch TensorBoard: tensorboard --logdir {args.profile_dir}")
        print(f"   2. View Chrome trace: Open trace_step_*.json in chrome://tracing")
        print(f"   3. Compare with DeepSpeed FLOPS: ./run_deepspeed_flops.sh --device 0 --num-steps 50")

    except Exception as e:
        print(f"Profiling analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

