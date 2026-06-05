#!/bin/bash
# Run TinyOpenFold V1 with PyTorch Profiler
# This script provides comprehensive profiling with detailed analysis

set -e

echo "========================================================================"
echo "TinyOpenFold V1 - PyTorch Profiler (Evoformer Analysis)"
echo "========================================================================"

# Default parameters
BATCH_SIZE=4
SEQ_LEN=64
NUM_SEQS=16
NUM_STEPS=20
PROFILE_STEPS=5
WARMUP_STEPS=3
PROFILE_DIR="./pytorch_profiles"
DEVICE=""
GENERATE_REPORT=""

# Parse arguments
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
        --num-seqs)
            NUM_SEQS="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --profile-steps)
            PROFILE_STEPS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --profile-dir)
            PROFILE_DIR="$2"
            shift 2
            ;;
        --generate-report)
            GENERATE_REPORT="--generate-report"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --batch-size <n>      Batch size (default: 4)"
            echo "  --seq-len <n>         Sequence length (default: 64)"
            echo "  --num-seqs <n>        Number of MSA sequences (default: 16)"
            echo "  --num-steps <n>       Total profiling steps (default: 20)"
            echo "  --profile-steps <n>   Active profiling steps (default: 5)"
            echo "  --device <n>          GPU device ID (e.g., 0, 1, 2)"
            echo "  --profile-dir <path>  Profile output directory (default: ./pytorch_profiles)"
            echo "  --generate-report     Generate comprehensive report"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create profile directory
mkdir -p "$PROFILE_DIR"

echo "Configuration:"
echo "   Batch size: $BATCH_SIZE"
echo "   Sequence length: $SEQ_LEN"
echo "   MSA sequences: $NUM_SEQS"
echo "   Total steps: $NUM_STEPS"
echo "   Profile steps: $PROFILE_STEPS"
echo "   Profile directory: $PROFILE_DIR"
if [ -n "$DEVICE" ]; then
    echo "   Device: GPU $DEVICE"
else
    echo "   Device: Default"
fi
echo ""

# Build command
CMD="python run_pytorch_profiler.py \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --num-seqs $NUM_SEQS \
    --num-steps $NUM_STEPS \
    --profile-steps $PROFILE_STEPS \
    --warmup-steps $WARMUP_STEPS \
    --profile-dir $PROFILE_DIR \
    --include-memory \
    --include-shapes"

if [ -n "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

if [ -n "$GENERATE_REPORT" ]; then
    CMD="$CMD $GENERATE_REPORT"
fi

# Run profiler
$CMD

echo ""
echo "========================================================================"
echo "PyTorch profiler analysis completed!"
echo "========================================================================"
echo "Profile data saved to: $PROFILE_DIR"
echo ""
echo "Visualization options:"
echo "  1. Chrome Trace Viewer (RECOMMENDED for timeline):"
echo "     - Open Chrome browser"
echo "     - Navigate to: chrome://tracing"
echo "     - Click 'Load' and select: $PROFILE_DIR/trace_step_*.json"
echo "     - Interactive timeline with kernel details"
echo ""
echo "  2. Comprehensive Report:"
echo "     less $PROFILE_DIR/comprehensive_profiling_report.md"
echo ""
echo "Analysis files:"
echo "  - comprehensive_profiling_report.md: Full analysis with recommendations"
echo "  - operator_analysis.json: Detailed operator performance"
echo "  - memory_analysis.json: Memory usage patterns"
echo "  - trace_step_*.json: Chrome trace format for chrome://tracing"
if [ -n "$GENERATE_REPORT" ]; then
    echo "  - comprehensive_profiling_report.md: Full analysis report"
fi
echo ""
echo "Compare with DeepSpeed FLOPS profiler:"
echo "  ./run_deepspeed_flops.sh --device 0 --num-steps 50"

