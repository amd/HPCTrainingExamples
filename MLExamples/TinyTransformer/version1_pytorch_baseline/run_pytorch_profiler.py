#!/usr/bin/env python3
"""
PyTorch Profiler Integration for Tiny LLaMA V1

This script provides enhanced PyTorch profiler integration with detailed analysis,
visualization, and bottleneck identification capabilities for the baseline model.

Features:
- Comprehensive profiler configuration
- Chrome trace export for detailed timeline analysis
- Operator-level performance breakdown
- Memory usage analysis
- Bottleneck identification and recommendations
- TensorBoard integration for visualization

Usage:
    # Run profiling with default settings
    python run_pytorch_profiler.py

    # Custom profiling configuration
    python run_pytorch_profiler.py --batch-size 16 --profile-steps 10

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
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import the model from tiny_llama_v1
from tiny_llama_v1 import TinyLlama, TinyLlamaConfig, SimpleTextDataset, setup_deterministic_environment


class PyTorchProfilerAnalyzer:
    """Advanced PyTorch profiler analysis and visualization."""

    def __init__(self, profile_dir: str):
        self.profile_dir = Path(profile_dir)
        self.profile_data = None
        self.analysis_results = {}

    def run_profiling(
        self,
        config: TinyLlamaConfig,
        batch_size: int = 8,
        num_steps: int = 20,
        warmup_steps: int = 3,
        profile_steps: int = 5,
        include_memory: bool = True,
        include_shapes: bool = True
    ) -> profile:
        """Run comprehensive PyTorch profiling session."""

        print(f"Starting PyTorch Profiler Analysis")
        print(f"   Profile directory: {self.profile_dir}")
        print(f"   Batch size: {batch_size}")
        print(f"   Total steps: {num_steps}")
        print(f"   Profile steps: {profile_steps}")
        print(f"   Memory profiling: {include_memory}")

        # Setup environment
        setup_deterministic_environment()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model and dataset
        model = TinyLlama(config).to(device)
        dataset = SimpleTextDataset(config.max_seq_len, config.vocab_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Ensure profile directory exists
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        # Configure profiler
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        def trace_handler(prof):
            """Custom trace handler for comprehensive output."""
            # Export Chrome trace
            chrome_trace_path = self.profile_dir / f"trace_step_{prof.step_num}.json"
            prof.export_chrome_trace(str(chrome_trace_path))

            # Export stacks (if available)
            if hasattr(prof, 'export_stacks'):
                stacks_path = self.profile_dir / f"stacks_step_{prof.step_num}.txt"
                prof.export_stacks(str(stacks_path), "self_cpu_time_total")

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

            for step in range(num_steps):
                # Get batch
                input_ids, labels = dataset.get_batch(batch_size)
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(input_ids, labels)
                loss = outputs['loss']

                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Profiler step
                prof.step()

                if step % 5 == 0:
                    print(f"   Step {step}/{num_steps}, Loss: {loss.item():.4f}")

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
        """Identify performance bottlenecks and optimization opportunities."""
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

        # Identify optimization targets
        optimization_targets = []
        for op in operator_data:
            name = op['name'].lower()

            # Attention-related optimizations
            if any(keyword in name for keyword in ['matmul', 'linear', 'addmm']):
                if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                    optimization_targets.append({
                        'operation': op['name'],
                        'optimization': 'QKV Fusion',
                        'potential_benefit': 'Reduce separate linear projections',
                        'priority': 'high'
                    })

            # SwiGLU optimizations
            if 'gate_proj' in name or 'up_proj' in name:
                optimization_targets.append({
                    'operation': op['name'],
                    'optimization': 'SwiGLU Fusion',
                    'potential_benefit': 'Combine gate and up projections',
                    'priority': 'medium'
                })

            # Softmax optimizations
            if 'softmax' in name:
                optimization_targets.append({
                    'operation': op['name'],
                    'optimization': 'Flash Attention',
                    'potential_benefit': 'Memory-efficient attention computation',
                    'priority': 'high'
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
        """Generate memory optimization recommendations."""
        recommendations = []

        # Check for high memory operations
        high_memory_ops = [event for event in memory_events if event['memory_usage'] > 1e6]  # > 1MB

        if high_memory_ops:
            recommendations.append(
                f"High memory operations detected: {len(high_memory_ops)} operations using >1MB. "
                "Consider gradient checkpointing or memory optimization techniques."
            )

        # Check for attention memory patterns
        attention_ops = [event for event in memory_events if 'attention' in event['name'].lower()]
        if attention_ops:
            recommendations.append(
                "Attention operations detected. Consider Flash Attention for memory-efficient computation."
            )

        # Check for temporary tensor creation
        temp_ops = [event for event in memory_events if event['count'] > 100]
        if temp_ops:
            recommendations.append(
                f"High-frequency operations detected: {len(temp_ops)} operations called >100 times. "
                "Consider tensor reuse or pre-allocation strategies."
            )

        return recommendations

    def generate_comprehensive_report(self, output_dir: str = None) -> str:
        """Generate comprehensive profiling report with recommendations."""
        if output_dir is None:
            output_dir = self.profile_dir

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_path = output_path / "comprehensive_profiling_report.md"

        report_content = f"""# PyTorch Profiler Analysis Report - Tiny LLaMA V1

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Profile Directory:** {self.profile_dir}

## Executive Summary

This report provides comprehensive performance analysis of the Tiny LLaMA V1 baseline implementation
using PyTorch's built-in profiler. The analysis focuses on identifying optimization opportunities
for subsequent versions of the workshop.

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

#### High Priority Optimizations

{self._format_optimization_recommendations('high')}

#### Medium Priority Optimizations

{self._format_optimization_recommendations('medium')}

## Next Steps for Version 2

Based on this analysis, the following optimizations should be implemented in Version 2:

1. **QKV Fusion**: Combine separate Q, K, V linear projections
2. **Flash Attention**: Implement memory-efficient attention computation
3. **SwiGLU Fusion**: Merge gate and up projections in feed-forward network
4. **Kernel Fusion**: Reduce number of separate GPU kernel launches

## Detailed Analysis Files

- **Operator Analysis**: `operator_analysis.json`
- **Memory Analysis**: `memory_analysis.json`
- **Chrome Traces**: `trace_step_*.json`
- **Performance Summary**: `performance_summary.json`

## Visualization

To visualize the profiling results:

1. **TensorBoard**: `tensorboard --logdir {self.profile_dir}`
2. **Chrome Trace**: Open `trace_step_*.json` in Chrome's chrome://tracing

---
*This report was generated by the Castille AI Workshop profiling tools.*
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
    parser = argparse.ArgumentParser(description='PyTorch Profiler for Tiny LLaMA V1')

    # Model configuration
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for profiling')
    parser.add_argument('--seq-len', type=int, default=128, help='Sequence length')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of layers')

    # Profiling configuration
    parser.add_argument('--num-steps', type=int, default=20, help='Total profiling steps')
    parser.add_argument('--warmup-steps', type=int, default=3, help='Warmup steps')
    parser.add_argument('--profile-steps', type=int, default=5, help='Active profiling steps')
    parser.add_argument('--profile-dir', type=str, default='./pytorch_profiles', help='Profile output directory')

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
    config = TinyLlamaConfig(
        hidden_dim=args.hidden_dim,
        n_layers=args.num_layers,
        max_seq_len=args.seq_len
    )

    print("PYTORCH PROFILER - TINY LLAMA V1 ANALYSIS")
    print("=" * 60)

    try:
        # Run profiling
        prof = analyzer.run_profiling(
            config=config,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
            include_memory=args.include_memory,
            include_shapes=args.include_shapes
        )

        # Analyze results
        print("\n" + "="*60)
        analyzer.analysis_results['operator_analysis'] = analyzer.analyze_operator_performance(prof)
        analyzer.analysis_results['memory_analysis'] = analyzer.analyze_memory_usage(prof)

        # Generate report
        if args.generate_report:
            report_path = analyzer.generate_comprehensive_report(args.output_dir)
            print(f"\nReport generated: {report_path}")

        print(f"\nProfiling analysis completed successfully!")
        print(f"Results saved to: {args.profile_dir}")
        print(f"Launch TensorBoard: tensorboard --logdir {args.profile_dir}")

    except Exception as e:
        print(f"Profiling analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()