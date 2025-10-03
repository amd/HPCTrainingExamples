#!/bin/bash
"""
ROCProfiler Analysis for Triton Kernels
Castille AI Workshop - Version 3

This script runs comprehensive ROCm profiling on Triton kernels
to analyze GPU utilization, memory patterns, and kernel efficiency.
"""

set -e

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/rocprof_results"
mkdir -p "$RESULTS_DIR"

echo "=== ROCProfiler Triton Kernel Analysis ==="
echo "Results will be saved to: $RESULTS_DIR"

# Function to run profiling with different configurations
run_triton_profiling() {
    local profile_name=$1
    local extra_args=$2
    local description=$3

    echo ""
    echo "Running $description..."
    echo "Profile: $profile_name"

    rocprof \
        --stats \
        --sys-trace \
        --hip-trace \
        --kernel-trace \
        --output-file "$RESULTS_DIR/${profile_name}.csv" \
        --timestamp on \
        --ctx-wait on \
        --ctx-switch on \
        $extra_args \
        python3 tiny_llama_v3.py 2>&1 | tee "$RESULTS_DIR/${profile_name}.log"

    echo "Completed: $profile_name"
}

# 1. Basic kernel profiling
echo ""
echo "1. Basic Triton Kernel Profiling"
run_triton_profiling "triton_basic" "" "Basic kernel execution analysis"

# 2. Memory access profiling
echo ""
echo "2. Memory Access Pattern Analysis"
run_triton_profiling "triton_memory" \
    "--hsa-trace --memory-trace" \
    "Memory access and bandwidth analysis"

# 3. Detailed kernel metrics
echo ""
echo "3. Detailed Kernel Metrics"
cat > "$RESULTS_DIR/kernel_metrics.txt" << 'EOF'
# Kernel-specific metrics for Triton analysis
pmc : Wavefronts VALUInsts SALUInsts SFetchInsts FlatVMemInsts LDSInsts FlatLDSInsts GDSInsts VALUUtilization FlatVMemUtilization MemUnitBusy L2CacheHit WriteUnitStalled ALUStalledByLDS LDSBankConflict
range: 0x1000000000000:0x2000000000000
gpu: 0
kernel: rmsnorm_kernel
kernel: swiglu_kernel
kernel: flash_attention_kernel
EOF

run_triton_profiling "triton_detailed" \
    "--input $RESULTS_DIR/kernel_metrics.txt" \
    "Detailed kernel performance counters"

# 4. Roofline analysis preparation
echo ""
echo "4. Roofline Analysis Data Collection"
cat > "$RESULTS_DIR/roofline_metrics.txt" << 'EOF'
# Metrics for roofline model analysis
pmc : GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_SMEM TCP_TOTAL_CACHE_ACCESSES_sum TCP_TOTAL_CACHE_ACCESSES_sum TCC_HIT_sum TCC_MISS_sum
range: 0x1000000000000:0x2000000000000
gpu: 0
EOF

run_triton_profiling "triton_roofline" \
    "--input $RESULTS_DIR/roofline_metrics.txt" \
    "Roofline model data collection"

# 5. Triton-specific kernel analysis
echo ""
echo "5. Triton Kernel Specific Analysis"
run_triton_profiling "triton_kernels" \
    "--kernel-trace --sys-trace" \
    "Triton kernel execution traces"

# 6. Compare with baseline (if available)
if [ -f "../version1_pytorch_baseline/rocprof_results/baseline.csv" ]; then
    echo ""
    echo "6. Comparing with Baseline Performance"

    python3 << 'EOF'
import pandas as pd
import sys
from pathlib import Path

try:
    # Load Triton results
    triton_file = Path("rocprof_results/triton_basic.csv")
    baseline_file = Path("../version1_pytorch_baseline/rocprof_results/baseline.csv")

    if triton_file.exists() and baseline_file.exists():
        triton_df = pd.read_csv(triton_file)
        baseline_df = pd.read_csv(baseline_file)

        print("=== Performance Comparison ===")

        if 'DurationNs' in triton_df.columns and 'DurationNs' in baseline_df.columns:
            triton_total = triton_df['DurationNs'].sum()
            baseline_total = baseline_df['DurationNs'].sum()
            speedup = baseline_total / triton_total if triton_total > 0 else 0

            print(f"Baseline total time: {baseline_total/1e6:.2f} ms")
            print(f"Triton total time: {triton_total/1e6:.2f} ms")
            print(f"Speedup: {speedup:.2f}x")

            # Save comparison
            with open("rocprof_results/comparison.txt", "w") as f:
                f.write(f"Baseline vs Triton Performance Comparison\n")
                f.write(f"========================================\n")
                f.write(f"Baseline total time: {baseline_total/1e6:.2f} ms\n")
                f.write(f"Triton total time: {triton_total/1e6:.2f} ms\n")
                f.write(f"Speedup: {speedup:.2f}x\n")
                f.write(f"Performance improvement: {((speedup-1)*100):.1f}%\n")
        else:
            print("Could not find duration columns for comparison")
    else:
        print("Missing profiling files for comparison")

except Exception as e:
    print(f"Error in comparison: {e}")
EOF
fi

# 7. Memory bandwidth analysis
echo ""
echo "7. Memory Bandwidth Analysis"
python3 << 'EOF'
import re
import pandas as pd
from pathlib import Path

def analyze_memory_bandwidth():
    print("=== Memory Bandwidth Analysis ===")

    results_dir = Path("rocprof_results")

    # Look for memory trace data
    for log_file in results_dir.glob("*.log"):
        print(f"\nAnalyzing: {log_file.name}")

        try:
            with open(log_file, 'r') as f:
                content = f.read()

            # Extract memory-related information
            memory_patterns = [
                r'Memory allocated: ([\d.]+) GB',
                r'Throughput: ([\d.]+) tokens/second',
                r'Average forward pass time: ([\d.]+) ms'
            ]

            for pattern in memory_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    metric = pattern.split(':')[0].replace('r\'', '').strip('(')
                    print(f"  {metric}: {matches[-1]}")

        except Exception as e:
            print(f"  Error reading {log_file}: {e}")

analyze_memory_bandwidth()
EOF

# 8. Kernel efficiency analysis
echo ""
echo "8. Kernel Efficiency Analysis"
python3 << 'EOF'
import pandas as pd
from pathlib import Path
import numpy as np

def analyze_kernel_efficiency():
    print("=== Kernel Efficiency Analysis ===")

    results_dir = Path("rocprof_results")
    csv_files = list(results_dir.glob("*.csv"))

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            print(f"\nAnalyzing: {csv_file.name}")

            if 'KernelName' in df.columns:
                # Group by kernel name
                kernel_groups = df.groupby('KernelName')

                for kernel_name, group in kernel_groups:
                    if any(triton_name in kernel_name.lower() for triton_name in
                          ['rmsnorm', 'swiglu', 'flash_attention']):

                        print(f"  Triton Kernel: {kernel_name}")

                        if 'DurationNs' in group.columns:
                            total_time = group['DurationNs'].sum()
                            avg_time = group['DurationNs'].mean()
                            call_count = len(group)

                            print(f"    Total time: {total_time/1e6:.3f} ms")
                            print(f"    Average time: {avg_time/1e6:.3f} ms")
                            print(f"    Call count: {call_count}")

                        # Look for utilization metrics
                        util_columns = [col for col in group.columns if 'util' in col.lower()]
                        for col in util_columns:
                            if not group[col].isna().all():
                                avg_util = group[col].mean()
                                print(f"    {col}: {avg_util:.1f}%")

        except Exception as e:
            print(f"Error analyzing {csv_file}: {e}")

analyze_kernel_efficiency()
EOF

# 9. Generate summary report
echo ""
echo "9. Generating Summary Report"
cat > "$RESULTS_DIR/triton_analysis_summary.md" << 'EOF'
# Triton Kernel ROCProfiler Analysis Summary

This report summarizes the ROCProfiler analysis of Triton kernels used in Version 3 of the workshop.

## Profiling Configurations

1. **Basic Profiling**: General kernel execution metrics
2. **Memory Analysis**: Memory access patterns and bandwidth utilization
3. **Detailed Metrics**: Comprehensive performance counters
4. **Roofline Data**: Data for roofline model analysis
5. **Kernel Traces**: Detailed execution traces
6. **Baseline Comparison**: Performance vs. PyTorch baseline

## Key Triton Kernels Analyzed

- `rmsnorm_kernel`: Custom RMSNorm implementation
- `swiglu_kernel`: Fused SwiGLU activation with projections
- `flash_attention_kernel`: Memory-efficient attention computation

## Analysis Files

- `triton_basic.csv`: Basic kernel profiling data
- `triton_memory.csv`: Memory access analysis
- `triton_detailed.csv`: Detailed performance counters
- `triton_roofline.csv`: Roofline model data
- `triton_kernels.csv`: Kernel execution traces
- `comparison.txt`: Performance comparison with baseline

## Usage

1. Review CSV files for detailed metrics
2. Use comparison.txt for performance improvements
3. Analyze memory patterns for optimization opportunities
4. Check kernel efficiency for utilization rates

## Next Steps

- Optimize memory access patterns based on analysis
- Tune block sizes and grid configurations
- Identify kernel fusion opportunities
- Compare with Version 4 (ultra-fused) results
EOF

echo ""
echo "=== ROCProfiler Analysis Complete ==="
echo ""
echo "Generated files:"
echo "  - rocprof_results/: All profiling data and logs"
echo "  - triton_analysis_summary.md: Summary report"
echo ""
echo "Key analysis points:"
echo "  1. Compare kernel execution times with baseline"
echo "  2. Check memory bandwidth utilization"
echo "  3. Analyze kernel efficiency metrics"
echo "  4. Identify optimization opportunities"
echo ""
echo "To view results:"
echo "  ls -la rocprof_results/"
echo "  cat rocprof_results/triton_analysis_summary.md"
echo "  cat rocprof_results/comparison.txt"