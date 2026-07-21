#!/bin/bash
#
# Performance Study: Compare TinyOpenFold V1, V2, and V3
#
# This script runs comprehensive performance comparisons across all three versions:
# - V1: PyTorch Baseline
# - V2: PyTorch Fused Operations
# - V3: Triton Custom Kernels
#
# Usage:
#   chmod +x launch_performance_study.sh
#   ./launch_performance_study.sh

echo "========================================================================="
echo "TinyOpenFold Performance Study: V1 vs V2 vs V3"
echo "========================================================================="
echo ""

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STUDY_DIR="performance_study_${TIMESTAMP}"
NUM_STEPS=30
BATCH_SIZE=4
SEQ_LEN=64
NUM_RUNS=3

echo "Study Configuration:"
echo "  Output directory: ${STUDY_DIR}"
echo "  Training steps: ${NUM_STEPS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Sequence length: ${SEQ_LEN}"
echo "  Runs per version: ${NUM_RUNS}"
echo ""

# Create study directory
mkdir -p ${STUDY_DIR}

# Save configuration
cat > ${STUDY_DIR}/config.json << EOF
{
  "timestamp": "${TIMESTAMP}",
  "num_steps": ${NUM_STEPS},
  "batch_size": ${BATCH_SIZE},
  "seq_len": ${SEQ_LEN},
  "num_runs": ${NUM_RUNS},
  "versions": ["v1_baseline", "v2_fused", "v3_triton"]
}
EOF

echo "Configuration saved to ${STUDY_DIR}/config.json"
echo ""

# =========================================================================
# Helper Functions
# =========================================================================

run_version() {
    local version=$1
    local version_dir=$2
    local script=$3
    local run=$4
    
    echo "-------------------------------------------"
    echo "Running ${version} (Run ${run}/${NUM_RUNS})"
    echo "-------------------------------------------"
    
    # Save current directory
    local current_dir=$(pwd)
    
    # Create output directory with absolute path
    local output_dir="${current_dir}/${STUDY_DIR}/${version}_run${run}"
    mkdir -p ${output_dir}
    
    cd ${version_dir}
    
    python3 ${script} \
        --batch-size ${BATCH_SIZE} \
        --seq-len ${SEQ_LEN} \
        --num-steps ${NUM_STEPS} \
        --num-blocks 4 \
        > ${output_dir}/output.log 2>&1
    
    local exit_code=$?
    
    # Copy performance summary if it exists
    if [ -f "pytorch_profiles/performance_summary.json" ]; then
        cp pytorch_profiles/performance_summary.json ${output_dir}/
    elif [ -f "pytorch_profiles_v2/performance_summary_v2.json" ]; then
        cp pytorch_profiles_v2/performance_summary_v2.json ${output_dir}/
    elif [ -f "triton_profiles/performance_summary_v3.json" ]; then
        cp triton_profiles/performance_summary_v3.json ${output_dir}/
    fi
    
    cd - > /dev/null
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ ${version} Run ${run} completed successfully"
    else
        echo "✗ ${version} Run ${run} failed (exit code: ${exit_code})"
    fi
    echo ""
    
    return $exit_code
}

# =========================================================================
# Run V1: PyTorch Baseline
# =========================================================================

echo "========================================================================="
echo "Version 1: PyTorch Baseline"
echo "========================================================================="
echo ""

V1_DIR="../version1_pytorch_baseline"
if [ -d "${V1_DIR}" ]; then
    for run in $(seq 1 ${NUM_RUNS}); do
        run_version "v1_baseline" "${V1_DIR}" "tiny_openfold_v1.py" ${run}
    done
else
    echo "✗ Version 1 directory not found: ${V1_DIR}"
    echo "  Skipping V1 benchmark"
    echo ""
fi

# =========================================================================
# Run V2: PyTorch Fused
# =========================================================================

echo "========================================================================="
echo "Version 2: PyTorch Fused Operations"
echo "========================================================================="
echo ""

V2_DIR="../version2_pytorch_fused"
if [ -d "${V2_DIR}" ]; then
    for run in $(seq 1 ${NUM_RUNS}); do
        run_version "v2_fused" "${V2_DIR}" "tiny_openfold_v2.py" ${run}
    done
else
    echo "✗ Version 2 directory not found: ${V2_DIR}"
    echo "  Skipping V2 benchmark"
    echo ""
fi

# =========================================================================
# Run V3: Triton Custom Kernels
# =========================================================================

echo "========================================================================="
echo "Version 3: Triton Custom Kernels"
echo "========================================================================="
echo ""

V3_DIR="."
for run in $(seq 1 ${NUM_RUNS}); do
    run_version "v3_triton" "${V3_DIR}" "tiny_openfold_v3.py" ${run}
done

# =========================================================================
# Analyze Results
# =========================================================================

echo "========================================================================="
echo "Analyzing Results"
echo "========================================================================="
echo ""

# Create Python analysis script
cat > ${STUDY_DIR}/analyze_results.py << 'ANALYSIS_SCRIPT'
#!/usr/bin/env python3
"""Analyze performance study results."""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_results(study_dir):
    """Load all performance results."""
    results = {}
    study_path = Path(study_dir)
    
    for version in ['v1_baseline', 'v2_fused', 'v3_triton']:
        results[version] = []
        
        for run_dir in sorted(study_path.glob(f'{version}_run*')):
            # Try different file names
            for filename in ['performance_summary.json', 'performance_summary_v2.json', 'performance_summary_v3.json']:
                json_file = run_dir / filename
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        results[version].append(data)
                    break
    
    return results

def compute_statistics(results):
    """Compute mean and std for each metric."""
    stats = {}
    
    for version, runs in results.items():
        if not runs:
            continue
        
        stats[version] = {}
        
        # Extract metrics from all runs
        metrics = {}
        for run in runs:
            perf = run.get('performance_summary', {})
            for key, value in perf.items():
                if isinstance(value, (int, float)):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        # Compute statistics (convert numpy types to Python native types for JSON)
        for metric, values in metrics.items():
            stats[version][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    return stats

def create_comparison_plots(stats, output_dir):
    """Create comparison plots."""
    output_path = Path(output_dir)
    
    # Training speed comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    versions = list(stats.keys())
    speeds = [stats[v]['avg_training_speed']['mean'] for v in versions if 'avg_training_speed' in stats[v]]
    errors = [stats[v]['avg_training_speed']['std'] for v in versions if 'avg_training_speed' in stats[v]]
    
    x = np.arange(len(versions))
    bars = ax.bar(x, speeds, yerr=errors, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    ax.set_xlabel('Version', fontsize=12)
    ax.set_ylabel('Training Speed (samples/sec)', fontsize=12)
    ax.set_title('TinyOpenFold Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['V1: Baseline', 'V2: Fused', 'V3: Triton'])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, speed) in enumerate(zip(bars, speeds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speed:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'performance_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path / 'performance_comparison.png'}")
    plt.close()
    
    # Memory usage comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    memory = [stats[v]['peak_memory_mb']['mean'] for v in versions if 'peak_memory_mb' in stats[v]]
    memory_errors = [stats[v]['peak_memory_mb']['std'] for v in versions if 'peak_memory_mb' in stats[v]]
    
    bars = ax.bar(x, memory, yerr=memory_errors, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    ax.set_xlabel('Version', fontsize=12)
    ax.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['V1: Baseline', 'V2: Fused', 'V3: Triton'])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mem) in enumerate(zip(bars, memory)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'memory_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path / 'memory_comparison.png'}")
    plt.close()

def generate_summary_report(stats, config, output_dir):
    """Generate markdown summary report."""
    output_path = Path(output_dir)
    
    with open(output_path / 'results_summary.md', 'w') as f:
        f.write('# TinyOpenFold Performance Study Results\n\n')
        f.write(f"**Study Date**: {config.get('timestamp', 'N/A')}\n\n")
        f.write(f"**Configuration**:\n")
        f.write(f"- Batch size: {config.get('batch_size', 'N/A')}\n")
        f.write(f"- Sequence length: {config.get('seq_len', 'N/A')}\n")
        f.write(f"- Training steps: {config.get('num_steps', 'N/A')}\n")
        f.write(f"- Runs per version: {config.get('num_runs', 'N/A')}\n\n")
        
        f.write('## Performance Summary\n\n')
        f.write('| Metric | V1 Baseline | V2 Fused | V3 Triton | V3 vs V1 |\n')
        f.write('|--------|-------------|----------|-----------|----------|\n')
        
        # Training speed
        v1_speed = stats.get('v1_baseline', {}).get('avg_training_speed', {}).get('mean', 0)
        v2_speed = stats.get('v2_fused', {}).get('avg_training_speed', {}).get('mean', 0)
        v3_speed = stats.get('v3_triton', {}).get('avg_training_speed', {}).get('mean', 0)
        
        speedup = v3_speed / v1_speed if v1_speed > 0 else 0
        
        f.write(f'| Training Speed (samples/s) | {v1_speed:.1f} | {v2_speed:.1f} | {v3_speed:.1f} | {speedup:.2f}x |\n')
        
        # Memory usage
        v1_mem = stats.get('v1_baseline', {}).get('peak_memory_mb', {}).get('mean', 0)
        v2_mem = stats.get('v2_fused', {}).get('peak_memory_mb', {}).get('mean', 0)
        v3_mem = stats.get('v3_triton', {}).get('peak_memory_mb', {}).get('mean', 0)
        
        mem_reduction = (v1_mem - v3_mem) / v1_mem * 100 if v1_mem > 0 else 0
        
        f.write(f'| Peak Memory (MB) | {v1_mem:.1f} | {v2_mem:.1f} | {v3_mem:.1f} | {mem_reduction:.1f}% reduction |\n')
        
        # Batch time
        v1_batch = stats.get('v1_baseline', {}).get('avg_batch_time', {}).get('mean', 0) * 1000
        v2_batch = stats.get('v2_fused', {}).get('avg_batch_time', {}).get('mean', 0) * 1000
        v3_batch = stats.get('v3_triton', {}).get('avg_batch_time', {}).get('mean', 0) * 1000
        
        f.write(f'| Batch Time (ms) | {v1_batch:.1f} | {v2_batch:.1f} | {v3_batch:.1f} | {v1_batch/v3_batch:.2f}x faster |\n')
        
        f.write('\n## Detailed Results\n\n')
        
        for version in ['v1_baseline', 'v2_fused', 'v3_triton']:
            if version not in stats:
                continue
            
            f.write(f'### {version.upper()}\n\n')
            f.write('| Metric | Mean | Std Dev | Min | Max |\n')
            f.write('|--------|------|---------|-----|-----|\n')
            
            for metric, values in stats[version].items():
                if metric == 'avg_training_speed':
                    f.write(f"| Training Speed (s/s) | {values['mean']:.2f} | {values['std']:.2f} | {values['min']:.2f} | {values['max']:.2f} |\n")
                elif metric == 'peak_memory_mb':
                    f.write(f"| Peak Memory (MB) | {values['mean']:.1f} | {values['std']:.1f} | {values['min']:.1f} | {values['max']:.1f} |\n")
                elif 'time' in metric.lower():
                    f.write(f"| {metric} (ms) | {values['mean']*1000:.2f} | {values['std']*1000:.2f} | {values['min']*1000:.2f} | {values['max']*1000:.2f} |\n")
            
            f.write('\n')
        
        f.write('## Key Findings\n\n')
        f.write(f'1. **Performance**: Version 3 achieves {speedup:.2f}x speedup over baseline\n')
        f.write(f'2. **Memory**: {mem_reduction:.1f}% reduction in peak memory usage\n')
        f.write(f'3. **Optimizations**: Triton custom kernels provide significant improvements\n')
        f.write('\n')
        f.write('## Plots\n\n')
        f.write('![Performance Comparison](performance_comparison.png)\n\n')
        f.write('![Memory Comparison](memory_comparison.png)\n\n')
    
    print(f"  Saved: {output_path / 'results_summary.md'}")

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <study_dir>")
        sys.exit(1)
    
    study_dir = sys.argv[1]
    
    print(f"Analyzing results from: {study_dir}")
    print("")
    
    # Load configuration
    config_file = Path(study_dir) / 'config.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load results
    print("Loading results...")
    results = load_results(study_dir)
    
    for version, runs in results.items():
        print(f"  {version}: {len(runs)} runs")
    print("")
    
    # Compute statistics
    print("Computing statistics...")
    stats = compute_statistics(results)
    
    # Save statistics
    stats_file = Path(study_dir) / 'statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_file}")
    print("")
    
    # Create plots
    print("Creating plots...")
    create_comparison_plots(stats, study_dir)
    print("")
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(stats, config, study_dir)
    print("")
    
    print("Analysis complete!")

if __name__ == '__main__':
    main()
ANALYSIS_SCRIPT

chmod +x ${STUDY_DIR}/analyze_results.py

# Run analysis
python3 ${STUDY_DIR}/analyze_results.py ${STUDY_DIR}

# =========================================================================
# Display Summary
# =========================================================================

echo "========================================================================="
echo "Performance Study Complete!"
echo "========================================================================="
echo ""
echo "Results saved in: ${STUDY_DIR}/"
echo ""
echo "Key files:"
echo "  - config.json: Study configuration"
echo "  - results_summary.md: Detailed analysis report"
echo "  - performance_comparison.png: Performance chart"
echo "  - memory_comparison.png: Memory usage chart"
echo "  - statistics.json: Statistical analysis"
echo ""
echo "To view the summary:"
echo "  cat ${STUDY_DIR}/results_summary.md"
echo ""

