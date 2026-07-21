#!/bin/bash
#
# ROCProfiler Integration for TinyOpenFold V3 Triton Kernels
#
# This script uses rocprofv3 to collect hardware-level metrics
# for Triton kernels running on AMD GPUs.
#
# Usage:
#   chmod +x run_rocprof_triton.sh
#   ./run_rocprof_triton.sh

echo "========================================="
echo "ROCProfiler for TinyOpenFold V3"
echo "Triton Kernel Hardware Profiling"
echo "========================================="
echo ""

# Configuration
OUTPUT_DIR="rocprof_results_v3"
PYTHON_SCRIPT="tiny_openfold_v3.py"
BATCH_SIZE=4
NUM_STEPS=20

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "Configuration:"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Python script: ${PYTHON_SCRIPT}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Training steps: ${NUM_STEPS}"
echo ""

# Check if rocprofv3 is available
if ! command -v rocprofv3 &> /dev/null; then
    echo "ERROR: rocprofv3 not found in PATH"
    echo "Please ensure ROCm tools are installed and in PATH."
    echo "Try: export PATH=\$PATH:/opt/rocm/bin"
    exit 1
fi

echo "ROCm version:"
# Try to get ROCm version from module system
if command -v module &> /dev/null; then
    ROCM_VERSION=$(module list 2>&1 | grep -oP 'rocm/\K[0-9.]+' | head -1)
    if [ -n "$ROCM_VERSION" ]; then
        echo "  rocm/$ROCM_VERSION"
    else
        echo "  rocm/6.4.1"
    fi
else
    # Fallback: use default version
    echo "  rocm/6.4.1"
fi
echo ""

# =========================================================================
# 1. Basic Kernel Timing
# =========================================================================
echo "========================================="
echo "1. Basic Kernel Timing"
echo "========================================="

# Generate output file prefix with timestamp
OUTPUT_PREFIX="$(hostname)_$$.$(date +%s%N)"

rocprofv3 \
    --kernel-trace \
    --stats \
    --output-format csv \
    --output-directory ${OUTPUT_DIR} \
    -o ${OUTPUT_PREFIX} \
    -- python3 ${PYTHON_SCRIPT} \
        --batch-size ${BATCH_SIZE} \
        --num-steps ${NUM_STEPS} \
    > ${OUTPUT_DIR}/kernel_timing.log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Kernel timing complete"
    echo "  Results: ${OUTPUT_DIR}/${OUTPUT_PREFIX}_kernel_stats.csv"
    echo "  Trace: ${OUTPUT_DIR}/${OUTPUT_PREFIX}_kernel_trace.csv"
    
    # Generate hotspot summary
    KERNEL_STATS="${OUTPUT_DIR}/${OUTPUT_PREFIX}_kernel_stats.csv"
    if [ -f "$KERNEL_STATS" ] && command -v python3 &> /dev/null; then
        echo ""
        echo "Hotspot Summary (Top 15 kernels by execution time):"
        echo "---------------------------------------------------"
        python3 - "$KERNEL_STATS" << 'PYEOF' 2>&1
import csv
import sys
from pathlib import Path
import re

def shorten_kernel_name(name, max_len=45):
    """Shorten kernel name for readability."""
    if len(name) <= max_len:
        return name
    
    # Try to extract meaningful parts
    # Remove common prefixes
    name = re.sub(r'^void\s+', '', name)
    name = re.sub(r'^__global__\s+', '', name)
    
    # If still too long, truncate intelligently
    if len(name) > max_len:
        # Try to keep the last part (function name)
        parts = name.split('::')
        if len(parts) > 1:
            # Keep last part and truncate middle
            last_part = parts[-1]
            if len(last_part) <= max_len - 10:
                return f"...{last_part}"
        # Simple truncation with ellipsis
        return name[:max_len-3] + "..."
    return name

if len(sys.argv) < 2:
    print("Error: Kernel stats file path not provided", file=sys.stderr)
    sys.exit(1)

kernel_stats = Path(sys.argv[1])
if not kernel_stats.exists():
    print(f"Error: Kernel stats file not found: {kernel_stats}", file=sys.stderr)
    sys.exit(1)

try:
    with open(kernel_stats, 'r') as f:
        reader = csv.DictReader(f)
        kernels = list(reader)
    
    if not kernels:
        print("No kernel data found")
        sys.exit(0)
    
    # Sort by total duration
    kernels.sort(key=lambda x: float(x.get('TotalDurationNs', 0)), reverse=True)
    
    # Calculate total time
    total_time_ns = sum(float(k.get('TotalDurationNs', 0)) for k in kernels)
    total_time_ms = total_time_ns / 1e6
    
    # Print top 15 kernels
    print(f"{'Rank':>5} {'Kernel Name':>48} {'Time (ms)':>12} {'%':>10} {'Calls':>8} {'Avg (μs)':>10}")
    print("-" * 95)
    
    for i, kernel in enumerate(kernels[:15], 1):
        name = kernel.get('Name', 'Unknown')
        short_name = shorten_kernel_name(name, 48)
        duration_ns = float(kernel.get('TotalDurationNs', 0))
        duration_ms = duration_ns / 1e6
        calls = int(kernel.get('Calls', 0))
        avg_us = (duration_ns / calls / 1000) if calls > 0 else 0
        percent = (duration_ns / total_time_ns * 100) if total_time_ns > 0 else 0
        
        print(f"{i:>5} {short_name:>48} {duration_ms:>12.2f} {percent:>6.1f}%  {calls:>8} {avg_us:>10.1f}")
        sys.stdout.flush()
    
    print("-" * 95)
    print(f"{'Total':>5} {'':>48} {total_time_ms:>12.2f} {'100.0':>7}%")
    sys.stdout.flush()
except Exception as e:
    print(f"Error processing kernel stats: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
PYEOF
        HOTSPOT_EXIT=$?
        if [ $HOTSPOT_EXIT -ne 0 ]; then
            echo "  Warning: Could not generate hotspot summary (exit code: $HOTSPOT_EXIT)"
        fi
    fi
else
    echo "✗ Kernel timing failed"
fi
echo ""

# =========================================================================
# 2. Runtime Trace Analysis
# =========================================================================
echo "========================================="
echo "2. Runtime Trace Analysis"
echo "========================================="

OUTPUT_PREFIX_RUNTIME="$(hostname)_$$_runtime.$(date +%s%N)"

rocprofv3 \
    --runtime-trace \
    --output-format csv \
    --output-directory ${OUTPUT_DIR} \
    -o ${OUTPUT_PREFIX_RUNTIME} \
    -- python3 ${PYTHON_SCRIPT} \
        --batch-size ${BATCH_SIZE} \
        --num-steps ${NUM_STEPS} \
    > ${OUTPUT_DIR}/runtime_trace.log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Runtime trace complete"
    echo "  Results: ${OUTPUT_DIR}/${OUTPUT_PREFIX_RUNTIME}_runtime_trace.csv"
    echo "  Runtime trace includes: HIP API, HSA API, memory operations, and more"
else
    echo "✗ Runtime trace failed"
fi
echo ""

# =========================================================================
# 3. Time Trace (pftrace format for Perfetto visualization)
# =========================================================================
echo "========================================="
echo "3. Time Trace (pftrace format)"
echo "========================================="

OUTPUT_PREFIX_PFTRACE="$(hostname)_$$_pftrace.$(date +%s%N)"

rocprofv3 \
    --runtime-trace \
    --output-format pftrace \
    --output-directory ${OUTPUT_DIR} \
    -o ${OUTPUT_PREFIX_PFTRACE} \
    -- python3 ${PYTHON_SCRIPT} \
        --batch-size ${BATCH_SIZE} \
        --num-steps ${NUM_STEPS} \
    > ${OUTPUT_DIR}/pftrace.log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Time trace complete"
    echo "  Results: ${OUTPUT_DIR}/${OUTPUT_PREFIX_PFTRACE}_results.pftrace"
    echo "  View in Perfetto: https://ui.perfetto.dev/"
    echo "  Upload the .pftrace file to visualize timeline"
    echo "  Runtime trace includes multiple relevant domains"
else
    echo "✗ Time trace failed"
fi
echo ""

# =========================================================================
# 4. Generate Summary Report
# =========================================================================
echo "========================================="
echo "4. Generating Summary Report"
echo "========================================="

cat > ${OUTPUT_DIR}/triton_analysis_summary.md << 'EOF'
# TinyOpenFold V3 Triton Kernel Profiling Summary

## Profiling Session

**Date**: $(date)
**Model Version**: V3 (Triton Custom Kernels)
**Hardware**: AMD MI300X

## Files Generated

1. `*_kernel_stats.csv` - Kernel execution statistics
2. `*_kernel_trace.csv` - Kernel execution trace
3. `*_runtime_trace.csv` - Runtime trace (includes HIP API, HSA API, memory operations, and more)
4. `*_results.pftrace` - Time trace in Perfetto format (for visualization)
5. `*.log` - Execution logs

## Analysis Steps

### 1. Kernel Statistics Analysis

```bash
# View top kernels by execution time
find ${OUTPUT_DIR} -name "*_kernel_stats.csv" -exec cat {} \; | sort -t',' -k2 -nr | head -20
```

### 2. Runtime Trace Analysis

The runtime trace includes multiple relevant domains:
- HIP API calls
- HSA API calls
- Memory operations
- Kernel dispatches
- Other runtime events

```bash
# Analyze runtime trace
find ${OUTPUT_DIR} -name "*_runtime_trace.csv" -exec head -20 {} \;
```

### 3. Triton Kernel Identification

Triton kernels will appear with names containing:
- `layernorm_kernel`
- `flash_attention_kernel`
- `triton_` prefix

### 4. Time Trace Visualization

The pftrace file uses runtime trace and can be visualized using Perfetto:

1. Open https://ui.perfetto.dev/ in your browser
2. Click "Open trace file"
3. Upload the `*_results.pftrace` file
4. Explore the timeline to see:
   - Runtime events across multiple domains
   - HIP API calls
   - HSA API calls
   - Memory operations
   - Kernel dispatches
   - System-level events
   - Overlaps and dependencies

## Key Metrics to Review

1. **Kernel Execution Time**: Total time spent in each kernel
2. **Launch Overhead**: Time between kernel launches
3. **Memory Bandwidth**: Achieved vs theoretical bandwidth
4. **Occupancy**: SM utilization percentage

## Comparison with Baseline

Compare these metrics with Version 1 and Version 2 results to validate
the performance improvements from Triton kernel optimizations.

EOF

echo "✓ Summary report generated"
echo "  Report: ${OUTPUT_DIR}/triton_analysis_summary.md"
echo ""

# =========================================================================
# 5. Display Summary
# =========================================================================
echo "========================================="
echo "Profiling Complete!"
echo "========================================="
echo ""
echo "Results saved in: ${OUTPUT_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Review ${OUTPUT_DIR}/triton_analysis_summary.md"
echo "  2. Analyze kernel statistics in ${OUTPUT_DIR}/*_kernel_stats.csv"
echo "  3. Visualize time traces: Upload ${OUTPUT_DIR}/*_results.pftrace to https://ui.perfetto.dev/"
echo "  4. Compare with V1/V2 baseline results"
echo ""
echo "To view kernel statistics:"
echo "  find ${OUTPUT_DIR} -name '*_kernel_stats.csv' -exec cat {} \; | column -t -s, | less -S"
echo ""
echo "To visualize time traces:"
echo "  1. Open https://ui.perfetto.dev/ in your browser"
echo "  2. Click 'Open trace file' and upload ${OUTPUT_DIR}/*_results.pftrace"
echo ""

