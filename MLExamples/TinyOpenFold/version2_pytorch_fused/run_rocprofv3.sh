#!/bin/bash

# rocprofv3 Profiling Integration for Tiny OpenFold V2
# This script provides comprehensive rocprofv3 profiling for kernel-level analysis

set -e  # Exit on error

# Save script directory for absolute path references
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

log_rocprof() {
    echo -e "${PURPLE}[ROCPROF]${NC} $1"
}

# Default configuration
BATCH_SIZE=4
SEQ_LEN=64
NUM_BLOCKS=4
NUM_SEQS=16
NUM_STEPS=20
OUTPUT_DIR="./rocprofv3_profiles_v2"
PROFILE_KERNELS=false
KERNELS_EXPLICITLY_SET=false
PROFILE_HIP_TRACE=false
TRACE_GPU_MEMORY=false
RUNTIME_TRACE=false
MARKER_TRACE=false
SYS_TRACE=false
TRUNCATE_KERNELS=false
DETAILED_METRICS=false
FUSION_ANALYSIS=true
OUTPUT_PFTRACE=false

# Fusion configuration
ENABLE_ALL_FUSION=true
DISABLE_FLASH=false
DISABLE_QKV=false
DISABLE_TRIANGLE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --num-blocks)
            NUM_BLOCKS="$2"
            shift 2
            ;;
        --num-seqs)
            NUM_SEQS="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --profile-kernels | -k)
            PROFILE_KERNELS=true
            KERNELS_EXPLICITLY_SET=true
            shift
            ;;
        --no-kernel-trace | -nk)
            PROFILE_KERNELS=false
            KERNELS_EXPLICITLY_SET=true
            shift
            ;;
        --profile-hip-trace | -ht)
            PROFILE_HIP_TRACE=true
            shift
            ;;
        --no-hip-trace | -nht)
            PROFILE_HIP_TRACE=false
            shift
            ;;
        --trace-gpu-memory | -m)
            TRACE_GPU_MEMORY=true
            shift
            ;;
        --runtime-trace | -r)
            RUNTIME_TRACE=true
            shift
            ;;
        --no-runtime-trace | -nr)
            RUNTIME_TRACE=false
            shift
            ;;
        --marker-trace | -mt)
            MARKER_TRACE=true
            shift
            ;;
        --no-marker-trace | -nmt)
            MARKER_TRACE=false
            shift
            ;;
        --sys-trace | -s)
            SYS_TRACE=true
            shift
            ;;
        --no-sys-trace | -ns)
            SYS_TRACE=false
            shift
            ;;
        --truncate-kernels | -tk)
            TRUNCATE_KERNELS=true
            shift
            ;;
        --no-truncate-kernels | -ntk)
            TRUNCATE_KERNELS=false
            shift
            ;;
        --detailed-metrics)
            DETAILED_METRICS=true
            shift
            ;;
        --output-pftrace | -pf)
            OUTPUT_PFTRACE=true
            shift
            ;;
        --no-pftrace | -npf)
            OUTPUT_PFTRACE=false
            shift
            ;;
        --no-fusion-analysis)
            FUSION_ANALYSIS=false
            shift
            ;;
        --disable-all-fusion)
            ENABLE_ALL_FUSION=false
            shift
            ;;
        --disable-flash)
            DISABLE_FLASH=true
            shift
            ;;
        --disable-qkv)
            DISABLE_QKV=true
            shift
            ;;
        --disable-triangle)
            DISABLE_TRIANGLE=true
            shift
            ;;
        --help|-h)
            echo "rocprofv3 Profiling for Tiny OpenFold V2"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --batch-size N          Batch size (default: 4)"
            echo "  --seq-len N             Sequence length (default: 64)"
            echo "  --num-blocks N          Number of Evoformer blocks (default: 4)"
            echo "  --num-seqs N            Number of MSA sequences (default: 16)"
            echo "  --num-steps N           Training steps (default: 20)"
            echo "  --output-dir DIR        Output directory"
            echo "  --profile-kernels       Enable kernel profiling (default)"
            echo "  --no-kernel-trace       Disable kernel tracing"
            echo "  --profile-hip-trace     Enable HIP API tracing"
            echo "  --no-hip-trace          Disable HIP API tracing"
            echo "  --trace-gpu-memory | -m Enable GPU memory tracing"
            echo "  --runtime-trace         Enable runtime trace"
            echo "  --no-runtime-trace      Disable runtime trace"
            echo "  --marker-trace | -mt    Enable marker trace"
            echo "  --no-marker-trace | -nmt Disable marker trace"
            echo "  --sys-trace | -s        Enable sys trace"
            echo "  --no-sys-trace | -ns    Disable sys trace"
            echo "  --truncate-kernels | -tk Enable kernel name truncation (default: disabled)"
            echo "  --no-truncate-kernels | -ntk Disable kernel name truncation"
            echo "  --detailed-metrics      Enable detailed hardware metrics"
            echo "  --output-pftrace | -pf Enable pftrace time trace output format"
            echo "  --no-pftrace | -npf     Disable pftrace output (default)"
            echo "  --no-fusion-analysis    Disable fusion-specific analysis"
            echo ""
            echo "Fusion Configuration:"
            echo "  --disable-all-fusion    Disable all fusions (baseline mode)"
            echo "  --disable-flash         Disable Flash Attention only"
            echo "  --disable-qkv           Disable QKV fusion only"
            echo "  --disable-triangle      Disable triangle fusion only"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Profile with all fusions"
            echo "  $0 --batch-size 8 --seq-len 128      # Larger workload"
            echo "  $0 --disable-all-fusion              # Baseline comparison"
            echo "  $0 --detailed-metrics                # Detailed hardware counters"
            echo "  $0 --output-pftrace                  # Generate pftrace time trace output"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Enable PROFILE_KERNELS by default only if no other trace options are enabled
if [ "$KERNELS_EXPLICITLY_SET" = false ]; then
    if [ "$PROFILE_HIP_TRACE" = false ] && [ "$TRACE_GPU_MEMORY" = false ] && [ "$RUNTIME_TRACE" = false ] && [ "$MARKER_TRACE" = false ] && [ "$SYS_TRACE" = false ]; then
        PROFILE_KERNELS=true
    fi
fi

# Check if rocprofv3 is available
if ! command -v rocprofv3 &> /dev/null; then
    log_error "rocprofv3 not found. Please ensure ROCm tools are installed and in PATH."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
log_info "======================================================================"
log_info "Tiny OpenFold V2 - rocprofv3 Profiling"
log_info "======================================================================"
echo ""
log_info "Configuration:"
log_info "  Batch size: $BATCH_SIZE"
log_info "  Sequence length: $SEQ_LEN"
log_info "  Evoformer blocks: $NUM_BLOCKS"
log_info "  MSA sequences: $NUM_SEQS"
log_info "  Training steps: $NUM_STEPS"
log_info "  Output directory: $OUTPUT_DIR"
echo ""
log_info "Profiling Options:"
log_info "  Kernel tracing: $PROFILE_KERNELS"
log_info "  Truncate kernels: $TRUNCATE_KERNELS"
log_info "  HIP API tracing: $PROFILE_HIP_TRACE"
log_info "  GPU memory tracing: $TRACE_GPU_MEMORY"
log_info "  Runtime trace: $RUNTIME_TRACE"
log_info "  Marker trace: $MARKER_TRACE"
log_info "  Sys trace: $SYS_TRACE"
log_info "  Detailed metrics: $DETAILED_METRICS"
log_info "  Pftrace output: $OUTPUT_PFTRACE"
log_info "  Fusion analysis: $FUSION_ANALYSIS"
echo ""
log_info "Fusion Configuration:"
log_info "  All fusions: $ENABLE_ALL_FUSION"
if [ "$ENABLE_ALL_FUSION" = false ]; then
    log_info "  Running in baseline mode (all fusions disabled)"
else
    log_info "  Flash Attention: $([ "$DISABLE_FLASH" = true ] && echo "disabled" || echo "enabled")"
    log_info "  QKV Fusion: $([ "$DISABLE_QKV" = true ] && echo "disabled" || echo "enabled")"
    log_info "  Triangle Fusion: $([ "$DISABLE_TRIANGLE" = true ] && echo "disabled" || echo "enabled")"
fi
echo ""

# Build rocprofv3 command
ROCPROF_CMD="rocprofv3"
ROCPROF_ARGS=""

# add stats option by default
ROCPROF_ARGS="$ROCPROF_ARGS --stats"

# Add kernel tracing
if [ "$PROFILE_KERNELS" = true ]; then
    ROCPROF_ARGS="$ROCPROF_ARGS --kernel-trace"
    if [ "$TRUNCATE_KERNELS" = true ]; then
        ROCPROF_ARGS="$ROCPROF_ARGS --truncate-kernels"
    fi
fi

# Add HIP API tracing
if [ "$PROFILE_HIP_TRACE" = true ]; then
    ROCPROF_ARGS="$ROCPROF_ARGS --hip-trace"
fi

# Add GPU memory tracing
if [ "$TRACE_GPU_MEMORY" = true ]; then
    ROCPROF_ARGS="$ROCPROF_ARGS --memory-copy-trace"
fi

# Add runtime trace --runtime-trace from command line option if provided
if [ "$RUNTIME_TRACE" = true ]; then
    ROCPROF_ARGS="$ROCPROF_ARGS --runtime-trace"
fi

# Add marker trace
if [ "$MARKER_TRACE" = true ]; then
    ROCPROF_ARGS="$ROCPROF_ARGS --marker-trace"
fi

# Add sys trace
if [ "$SYS_TRACE" = true ]; then
    ROCPROF_ARGS="$ROCPROF_ARGS --sys-trace"
fi

# Add output format - default to csv if OUTPUT_PFTRACE is not set
if [ "$OUTPUT_PFTRACE" = true ]; then
    ROCPROF_ARGS="$ROCPROF_ARGS --output-format pftrace"
else
    ROCPROF_ARGS="$ROCPROF_ARGS --output-format csv"
fi

# Add output file prefix for rocprofv3 -o flag (similar to PyTorch profiler format: hostname_pid.timestamp)
# Format: {hostname}_{pid}.{nanoseconds_since_epoch}
# Use Python to get nanosecond timestamp (fallback to date if Python unavailable)
if command -v python3 &> /dev/null; then
    NANOSECONDS=$(python3 -c 'import time; print(int(time.time() * 1e9))' 2>/dev/null)
else
    # Fallback: use date with nanoseconds if available, otherwise seconds
    NANOSECONDS=$(date +%s%N 2>/dev/null || date +%s)000000000
fi
OUTPUT_FILE_PREFIX="$(hostname)_$$.${NANOSECONDS}"
ROCPROF_ARGS="$ROCPROF_ARGS -o $OUTPUT_FILE_PREFIX"

# Build Python command with absolute path
PYTHON_SCRIPT="$SCRIPT_DIR/tiny_openfold_v2.py"
PYTHON_ARGS="--batch-size $BATCH_SIZE --seq-len $SEQ_LEN --num-blocks $NUM_BLOCKS --num-seqs $NUM_SEQS --num-steps $NUM_STEPS"

# Add fusion configuration
if [ "$ENABLE_ALL_FUSION" = false ]; then
    PYTHON_ARGS="$PYTHON_ARGS --disable-all-fusion"
else
    [ "$DISABLE_FLASH" = true ] && PYTHON_ARGS="$PYTHON_ARGS --disable-flash-attention"
    [ "$DISABLE_QKV" = true ] && PYTHON_ARGS="$PYTHON_ARGS --disable-qkv-fusion-msa --disable-qkv-fusion-triangle"
    [ "$DISABLE_TRIANGLE" = true ] && PYTHON_ARGS="$PYTHON_ARGS --disable-triangle-fusion"
fi

# Run profiling
log_step "Starting rocprofv3 profiling..."
log_rocprof "Command: $ROCPROF_CMD $ROCPROF_ARGS -- python $PYTHON_SCRIPT $PYTHON_ARGS"
echo ""

cd "$OUTPUT_DIR"
$ROCPROF_CMD $ROCPROF_ARGS -- python "$PYTHON_SCRIPT" $PYTHON_ARGS 2>&1 | tee rocprofv3.log
cd - > /dev/null

log_step "Profiling complete!"

# Analyze results
log_step "Analyzing profiling results..."

# Find kernel stats file
KERNEL_STATS=$(find "$OUTPUT_DIR" -name "*_kernel_stats.csv" | head -n 1)

if [ -f "$KERNEL_STATS" ]; then
    log_info "Kernel statistics found: $KERNEL_STATS"
    
    # Generate summary report
    SUMMARY_FILE="$OUTPUT_DIR/rocprofv3_summary.txt"
    
    {
        echo "======================================================================"
        echo "Tiny OpenFold V2 - rocprofv3 Summary"
        echo "======================================================================"
        echo ""
        echo "Configuration:"
        echo "  Batch size: $BATCH_SIZE"
        echo "  Sequence length: $SEQ_LEN"
        echo "  Evoformer blocks: $NUM_BLOCKS"
        echo "  MSA sequences: $NUM_SEQS"
        echo "  Training steps: $NUM_STEPS"
        echo ""
        echo "Fusion Configuration:"
        echo "  All fusions: $ENABLE_ALL_FUSION"
        echo ""
        echo "Top GPU Kernels by Time:"
        echo "----------------------------------------------------------------------"
        
        # Parse and display top kernels
        if command -v python3 &> /dev/null; then
            python3 << EOF "$KERNEL_STATS"
import csv
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Error: Kernel stats file path not provided")
    sys.exit(1)

kernel_stats = Path(sys.argv[1])
if kernel_stats.exists():
    with open(kernel_stats, 'r') as f:
        reader = csv.DictReader(f)
        kernels = list(reader)
        
    # Sort by total duration
    kernels.sort(key=lambda x: float(x.get('TotalDurationNs', 0)), reverse=True)
    
    # Print top 20 kernels
    print(f"{'Rank':<6} {'Kernel Name':<50} {'Duration (ms)':<15} {'Calls':<10} {'Avg (us)':<12}")
    print("-" * 100)
    
    for i, kernel in enumerate(kernels[:20], 1):
        name = kernel.get('Name', 'Unknown')[:50]
        duration_ns = float(kernel.get('TotalDurationNs', 0))
        duration_ms = duration_ns / 1e6
        calls = int(kernel.get('Calls', 0))
        avg_us = (duration_ns / calls / 1000) if calls > 0 else 0
        print(f"{i:<6} {name:<50} {duration_ms:<15.2f} {calls:<10} {avg_us:<12.2f}")
    
    # Calculate total time
    total_time_ms = sum(float(k.get('TotalDurationNs', 0)) for k in kernels) / 1e6
    print("-" * 100)
    print(f"Total GPU Time: {total_time_ms:.2f} ms")
    
    # Fusion-specific analysis
    print("\n\nFusion-Specific Kernel Analysis:")
    print("-" * 100)
    
    fusion_categories = {
        'MSA Attention': ['msa', 'attention', 'qkv'],
        'Triangle Operations': ['triangle', 'einsum'],
        'Flash Attention': ['flash', 'scaled_dot'],
        'Memory Operations': ['memcpy', 'memset'],
    }
    
    for category, keywords in fusion_categories.items():
        category_kernels = [k for k in kernels if any(kw in k.get('Name', '').lower() for kw in keywords)]
        if category_kernels:
            cat_time_ms = sum(float(k.get('TotalDurationNs', 0)) for k in category_kernels) / 1e6
            cat_calls = sum(int(k.get('Calls', 0)) for k in category_kernels)
            cat_percent = (cat_time_ms / total_time_ms * 100) if total_time_ms > 0 else 0
            print(f"{category:<25} {cat_time_ms:>10.2f} ms ({cat_percent:>5.1f}%)  {cat_calls:>8} calls")
else:
    print(f"Error: Kernel stats file not found: {kernel_stats}")
EOF
        fi
        
        echo ""
        echo "======================================================================"
        
    } > "$SUMMARY_FILE"
    
    cat "$SUMMARY_FILE"
    log_info "Summary saved to: $SUMMARY_FILE"
else
    log_warning "Kernel statistics file not found"
fi

# List output files
echo ""
log_info "======================================================================"
log_info "Output Files:"
log_info "======================================================================"
ls -lh "$OUTPUT_DIR" | tail -n +2
echo ""

log_info "======================================================================"
log_info "rocprofv3 Profiling Complete!"
log_info "======================================================================"
echo ""
log_info "Results directory: $OUTPUT_DIR"
echo ""
log_info "Key files:"
log_info "  - rocprofv3.log              : Full profiling log"
log_info "  - *_kernel_stats.csv         : Kernel statistics"
log_info "  - rocprofv3_summary.txt      : Analysis summary"
if [ "$OUTPUT_PFTRACE" = true ]; then
    log_info "  - *.pftrace                 : Time trace output (pftrace format)"
fi
echo ""
log_info "To view kernel statistics:"
log_info "  less $OUTPUT_DIR/rocprofv3_summary.txt"
echo ""
log_info "To analyze CSV data:"
log_info "  python -c 'import pandas as pd; df = pd.read_csv(\"$KERNEL_STATS\"); print(df.head())'"
echo ""

# Cleanup
log_info "Profiling session complete!"


