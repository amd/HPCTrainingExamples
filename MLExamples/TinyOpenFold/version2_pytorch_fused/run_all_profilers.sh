#!/bin/bash

# Comprehensive Profiling Suite for Tiny OpenFold V2
# Runs all available profilers: PyTorch, ROCm tools, and generates comparative analysis

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
log_profiler() { echo -e "${PURPLE}[PROFILER]${NC} $1"; }

# Default configuration
BATCH_SIZE=4
SEQ_LEN=64
NUM_BLOCKS=4
NUM_SEQS=16
NUM_STEPS=30
OUTPUT_DIR="./complete_profiling_$(date +%Y%m%d_%H%M%S)"
ENABLE_ALL_FUSION=true
DEVICE=0

# Profiler selection
RUN_PYTORCH=true
RUN_ROCPROFV3=true
RUN_ROCPROF_SYS=true
RUN_ROCPROF_COMPUTE=true
QUICK_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --seq-len) SEQ_LEN="$2"; shift 2 ;;
        --num-blocks) NUM_BLOCKS="$2"; shift 2 ;;
        --num-seqs) NUM_SEQS="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --disable-all-fusion) ENABLE_ALL_FUSION=false; shift ;;
        --pytorch-only) RUN_ROCPROFV3=false; RUN_ROCPROF_SYS=false; RUN_ROCPROF_COMPUTE=false; shift ;;
        --rocm-only) RUN_PYTORCH=false; shift ;;
        --quick) QUICK_MODE=true; shift ;;
        --no-pytorch) RUN_PYTORCH=false; shift ;;
        --no-rocprofv3) RUN_ROCPROFV3=false; shift ;;
        --no-rocprof-sys) RUN_ROCPROF_SYS=false; shift ;;
        --no-rocprof-compute) RUN_ROCPROF_COMPUTE=false; shift ;;
        --help|-h)
            echo "Comprehensive Profiling Suite for Tiny OpenFold V2"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --batch-size N          Batch size (default: 4)"
            echo "  --seq-len N             Sequence length (default: 64)"
            echo "  --num-blocks N          Number of Evoformer blocks (default: 4)"
            echo "  --num-seqs N            Number of MSA sequences (default: 16)"
            echo "  --num-steps N           Training steps (default: 30)"
            echo "  --output-dir DIR        Output directory"
            echo "  --device N              GPU device (default: 0)"
            echo "  --disable-all-fusion    Disable all fusions"
            echo ""
            echo "Profiler Selection:"
            echo "  --pytorch-only          Run only PyTorch profiler"
            echo "  --rocm-only             Run only ROCm profilers"
            echo "  --no-pytorch            Skip PyTorch profiler"
            echo "  --no-rocprofv3          Skip rocprofv3"
            echo "  --no-rocprof-sys        Skip rocprof-sys"
            echo "  --no-rocprof-compute    Skip rocprof-compute"
            echo "  --quick                 Quick mode (reduced profiling steps)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all profilers"
            echo "  $0 --pytorch-only                     # PyTorch profiler only"
            echo "  $0 --quick                            # Quick profiling"
            echo "  $0 --disable-all-fusion               # Profile baseline"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Adjust for quick mode
if [ "$QUICK_MODE" = true ]; then
    NUM_STEPS=15
    RUN_ROCPROF_SYS=false  # Skip slowest profiler
    log_info "Quick mode enabled: reduced steps, skipping rocprof-sys"
fi

mkdir -p "$OUTPUT_DIR"

log_info "======================================================================"
log_info "Tiny OpenFold V2 - Comprehensive Profiling Suite"
log_info "======================================================================"
echo ""
log_info "Configuration:"
log_info "  Batch size: $BATCH_SIZE"
log_info "  Sequence length: $SEQ_LEN"
log_info "  Evoformer blocks: $NUM_BLOCKS"
log_info "  MSA sequences: $NUM_SEQS"
log_info "  Training steps: $NUM_STEPS"
log_info "  All fusions: $ENABLE_ALL_FUSION"
log_info "  Device: $DEVICE"
log_info "  Output directory: $OUTPUT_DIR"
echo ""
log_info "Profilers to run:"
[ "$RUN_PYTORCH" = true ] && log_info "  ✓ PyTorch Profiler"
[ "$RUN_ROCPROFV3" = true ] && log_info "  ✓ rocprofv3"
[ "$RUN_ROCPROF_SYS" = true ] && log_info "  ✓ rocprof-sys"
[ "$RUN_ROCPROF_COMPUTE" = true ] && log_info "  ✓ rocprof-compute"
echo ""

# Build common arguments
COMMON_ARGS="--batch-size $BATCH_SIZE --seq-len $SEQ_LEN --num-blocks $NUM_BLOCKS --num-seqs $NUM_SEQS --num-steps $NUM_STEPS --device $DEVICE"
[ "$ENABLE_ALL_FUSION" = false ] && COMMON_ARGS="$COMMON_ARGS --disable-all-fusion"

# Track profiling times
PROFILE_START=$(date +%s)

# 1. PyTorch Profiler
if [ "$RUN_PYTORCH" = true ]; then
    log_step "Running PyTorch Profiler (1/4)..."
    PYTORCH_DIR="$OUTPUT_DIR/pytorch_profiling"
    
    if [ -f "./run_pytorch_profiler.py" ]; then
        python run_pytorch_profiler.py $COMMON_ARGS --profile-dir $PYTORCH_DIR
        log_info "✓ PyTorch profiling complete"
    else
        log_warning "run_pytorch_profiler.py not found, skipping"
    fi
    echo ""
fi

# 2. rocprofv3
if [ "$RUN_ROCPROFV3" = true ]; then
    log_step "Running rocprofv3 (2/4)..."
    ROCPROFV3_DIR="$OUTPUT_DIR/rocprofv3_profiling"
    
    if [ -f "./run_rocprofv3.sh" ]; then
        ./run_rocprofv3.sh $COMMON_ARGS --output-dir $ROCPROFV3_DIR
        log_info "✓ rocprofv3 profiling complete"
    else
        log_warning "run_rocprofv3.sh not found, skipping"
    fi
    echo ""
fi

# 3. rocprof-sys
if [ "$RUN_ROCPROF_SYS" = true ]; then
    log_step "Running rocprof-sys (3/4)..."
    ROCPROF_SYS_DIR="$OUTPUT_DIR/rocprof_sys_profiling"
    
    if [ -f "./run_rocprof_sys.sh" ]; then
        ./run_rocprof_sys.sh $COMMON_ARGS --output-dir $ROCPROF_SYS_DIR
        log_info "✓ rocprof-sys profiling complete"
    else
        log_warning "run_rocprof_sys.sh not found, skipping"
    fi
    echo ""
fi

# 4. rocprof-compute
if [ "$RUN_ROCPROF_COMPUTE" = true ]; then
    log_step "Running rocprof-compute (4/4)..."
    
    if [ -f "./run_rocprof_compute.sh" ]; then
        cd "$OUTPUT_DIR"
        ../run_rocprof_compute.sh $COMMON_ARGS --output-name tinyfold_complete
        cd - > /dev/null
        log_info "✓ rocprof-compute profiling complete"
    else
        log_warning "run_rocprof_compute.sh not found, skipping"
    fi
    echo ""
fi

PROFILE_END=$(date +%s)
TOTAL_TIME=$((PROFILE_END - PROFILE_START))

# Generate summary report
log_step "Generating comprehensive summary..."

SUMMARY_FILE="$OUTPUT_DIR/PROFILING_SUMMARY.md"

cat > "$SUMMARY_FILE" << EOF
# Tiny OpenFold V2 - Comprehensive Profiling Summary

Generated: $(date '+%Y-%m-%d %H:%M:%S')

## Configuration

- Batch size: $BATCH_SIZE
- Sequence length: $SEQ_LEN
- Evoformer blocks: $NUM_BLOCKS
- MSA sequences: $NUM_SEQS
- Training steps: $NUM_STEPS
- All fusions enabled: $ENABLE_ALL_FUSION
- Device: $DEVICE
- Total profiling time: $TOTAL_TIME seconds

## Profiling Results

EOF

# Add results from each profiler
if [ "$RUN_PYTORCH" = true ] && [ -d "$PYTORCH_DIR" ]; then
    cat >> "$SUMMARY_FILE" << EOF
### PyTorch Profiler

Directory: \`$PYTORCH_DIR\`

**Key Files:**
- comprehensive_profiling_report.md - Detailed analysis
- fusion_analysis.json - Fusion statistics
- *.pt.trace.json - Chrome trace files

**View Results:**
\`\`\`bash
# View report
less $PYTORCH_DIR/comprehensive_profiling_report.md

# TensorBoard
tensorboard --logdir $PYTORCH_DIR

# Chrome trace
# Open chrome://tracing and load trace file
\`\`\`

EOF
fi

if [ "$RUN_ROCPROFV3" = true ] && [ -d "$ROCPROFV3_DIR" ]; then
    cat >> "$SUMMARY_FILE" << EOF
### rocprofv3

Directory: \`$ROCPROFV3_DIR\`

**Key Files:**
- rocprofv3_summary.txt - Kernel statistics summary
- *_kernel_stats.csv - Detailed kernel data

**View Results:**
\`\`\`bash
less $ROCPROFV3_DIR/rocprofv3_summary.txt
\`\`\`

EOF
fi

if [ "$RUN_ROCPROF_SYS" = true ] && [ -d "$ROCPROF_SYS_DIR" ]; then
    cat >> "$SUMMARY_FILE" << EOF
### rocprof-sys

Directory: \`$ROCPROF_SYS_DIR\`

**Key Files:**
- *.proto - Perfetto timeline trace

**View Results:**
1. Copy .proto file to local machine
2. Open https://ui.perfetto.dev
3. Load the .proto file

EOF
fi

if [ "$RUN_ROCPROF_COMPUTE" = true ]; then
    cat >> "$SUMMARY_FILE" << EOF
### rocprof-compute

Directory: \`$OUTPUT_DIR\`

**Key Files:**
- roofline_*.pdf - Roofline plots
- workloads/tinyfold_complete/ - Detailed metrics

**View Results:**
\`\`\`bash
# View roofline
open roofline_*.pdf

# List dispatches
cd $OUTPUT_DIR
rocprof-compute analyze -p workloads/tinyfold_complete/* --list-stats
\`\`\`

EOF
fi

cat >> "$SUMMARY_FILE" << EOF
## Analysis Recommendations

1. **Start with PyTorch Profiler** for high-level understanding
   - Identify hotspot operations
   - Analyze fusion impact

2. **Use rocprofv3** for kernel-level analysis
   - Check kernel execution times
   - Verify fusion effectiveness

3. **Use rocprof-sys** for timeline analysis
   - Identify synchronization issues
   - Check CPU-GPU overlaps

4. **Use rocprof-compute** for hardware utilization
   - Check memory bandwidth utilization
   - Analyze compute vs memory bound

## Next Steps

- Compare with baseline (V1) results
- Run ablation studies for individual fusions
- Optimize identified bottlenecks
- Test different batch sizes and sequence lengths

EOF

log_info "Summary report generated: $SUMMARY_FILE"

# Display summary
echo ""
log_info "======================================================================"
log_info "Comprehensive Profiling Complete!"
log_info "======================================================================"
echo ""
log_info "Results directory: $OUTPUT_DIR"
log_info "Total profiling time: $TOTAL_TIME seconds"
echo ""
log_info "Quick access:"
echo ""
[ "$RUN_PYTORCH" = true ] && log_info "  PyTorch: less $PYTORCH_DIR/comprehensive_profiling_report.md"
[ "$RUN_ROCPROFV3" = true ] && log_info "  rocprofv3: less $ROCPROFV3_DIR/rocprofv3_summary.txt"
[ "$RUN_ROCPROF_SYS" = true ] && log_info "  rocprof-sys: open https://ui.perfetto.dev (load .proto file)"
[ "$RUN_ROCPROF_COMPUTE" = true ] && log_info "  rocprof-compute: open $OUTPUT_DIR/roofline_*.pdf"
echo ""
log_info "  Summary: less $SUMMARY_FILE"
echo ""
log_info "======================================================================"


