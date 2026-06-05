#!/bin/bash
################################################################################
# TinyOpenFold V1 - DeepSpeed FLOPS Profiler
#
# This script runs comprehensive FLOPS analysis for the Evoformer architecture
# using DeepSpeed's FLOPS profiler to measure computational efficiency.
#
# Usage:
#   ./run_deepspeed_flops.sh [OPTIONS]
#
# Options:
#   --batch-size <n>        Batch size for profiling (default: 4)
#   --seq-len <n>           Sequence length (default: 64)
#   --num-seqs <n>          Number of MSA sequences (default: 16)
#   --num-steps <n>         Number of profiling steps (default: 10)
#   --device <n>            Specific GPU device ID to use (e.g., 0, 1, 2)
#   --multi-gpu             Profile across all available GPUs
#   --devices <ids>         Comma-separated GPU IDs (e.g., "0,1,2")
#   --output-dir <path>     Output directory (default: ./flops_analysis)
#   --detailed              Enable detailed FLOPS breakdown
#   --roofline              Generate roofline analysis data
#   --intensity             Analyze computational intensity
#   --all                   Run all analysis types
#   --help                  Show this help message
#
# Examples:
#   # Basic FLOPS profiling (single GPU, default device)
#   ./run_deepspeed_flops.sh
#
#   # Profile on specific GPU
#   ./run_deepspeed_flops.sh --device 1
#
#   # Multi-GPU profiling (all available GPUs - 8 on MI250X node)
#   ./run_deepspeed_flops.sh --multi-gpu
#
#   # Multi-GPU profiling (specific GPUs - all 8 on MI250X)
#   ./run_deepspeed_flops.sh --devices "0,1,2,3,4,5,6,7"
#
#   # Comprehensive analysis with all features
#   ./run_deepspeed_flops.sh --all --batch-size 8
#
#   # Custom configuration
#   ./run_deepspeed_flops.sh --seq-len 128 --num-blocks 8 --roofline
#
################################################################################

set -e

# Default configuration
BATCH_SIZE=4
SEQ_LEN=64
NUM_SEQS=16
MSA_DIM=64
PAIR_DIM=128
NUM_BLOCKS=4
NUM_STEPS=10
OUTPUT_DIR="./flops_analysis"
DEVICE=""
MULTI_GPU=""
DEVICES=""
DETAILED=""
ROOFLINE=""
INTENSITY=""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

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
        --msa-dim)
            MSA_DIM="$2"
            shift 2
            ;;
        --pair-dim)
            PAIR_DIM="$2"
            shift 2
            ;;
        --num-blocks)
            NUM_BLOCKS="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --multi-gpu)
            MULTI_GPU="--multi-gpu"
            shift
            ;;
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --detailed)
            DETAILED="--detailed-analysis"
            shift
            ;;
        --roofline)
            ROOFLINE="--generate-roofline"
            shift
            ;;
        --intensity)
            INTENSITY="--computational-intensity"
            shift
            ;;
        --all)
            DETAILED="--detailed-analysis"
            ROOFLINE="--generate-roofline"
            INTENSITY="--computational-intensity"
            shift
            ;;
        --help)
            grep "^#" "$0" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo -e "${CYAN}TinyOpenFold V1 - DeepSpeed FLOPS Profiler${NC}"
echo "                 Evoformer Architecture Analysis"
echo "========================================================================"
echo ""

# Check if DeepSpeed is available
if ! python3 -c "import deepspeed" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Warning: DeepSpeed not installed${NC}"
    echo "   The script will provide FLOPS estimates but detailed profiling requires DeepSpeed"
    echo ""
    echo "   To install DeepSpeed:"
    echo "   pip install deepspeed"
    echo ""
    read -p "Continue without DeepSpeed? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo -e "${BLUE}Configuration:${NC}"
echo "   Batch size: $BATCH_SIZE"
echo "   Sequence length: $SEQ_LEN"
echo "   MSA sequences: $NUM_SEQS"
echo "   MSA dimension: $MSA_DIM"
echo "   Pair dimension: $PAIR_DIM"
echo "   Evoformer blocks: $NUM_BLOCKS"
echo "   Profiling steps: $NUM_STEPS"
echo "   Output directory: $OUTPUT_DIR"

# Print device configuration
if [ -n "$MULTI_GPU" ]; then
    echo "   Mode: Multi-GPU (all available GPUs)"
elif [ -n "$DEVICES" ]; then
    echo "   Mode: Multi-GPU (GPUs: $DEVICES)"
elif [ -n "$DEVICE" ]; then
    echo "   Mode: Single GPU (device $DEVICE)"
else
    echo "   Mode: Single GPU (default device)"
fi
echo ""

# Check for GPU
if command -v rocm-smi &> /dev/null; then
    echo -e "${GREEN}AMD GPU detected:${NC}"
    rocm-smi --showproductname 2>/dev/null | grep "Card series" || echo "   ROCm available"
elif command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
else
    echo -e "${YELLOW}⚠️  No GPU detected, will use CPU (slow)${NC}"
fi
echo ""

# Run FLOPS profiling
echo -e "${GREEN}Starting FLOPS profiling...${NC}"
echo "========================================================================"
echo ""

# Build device arguments
DEVICE_ARGS=""
if [ -n "$MULTI_GPU" ]; then
    DEVICE_ARGS="$MULTI_GPU"
elif [ -n "$DEVICES" ]; then
    DEVICE_ARGS="--devices $DEVICES"
elif [ -n "$DEVICE" ]; then
    DEVICE_ARGS="--device $DEVICE"
fi

python3 run_deepspeed_flops.py \
    --batch-size "$BATCH_SIZE" \
    --seq-len "$SEQ_LEN" \
    --num-seqs "$NUM_SEQS" \
    --msa-dim "$MSA_DIM" \
    --pair-dim "$PAIR_DIM" \
    --num-blocks "$NUM_BLOCKS" \
    --num-steps "$NUM_STEPS" \
    --output-dir "$OUTPUT_DIR" \
    $DEVICE_ARGS \
    $DETAILED \
    $ROOFLINE \
    $INTENSITY

EXIT_CODE=$?

echo ""
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ DeepSpeed FLOPS profiler completed successfully!${NC}"
    echo ""
    echo -e "${CYAN}Results saved to: ${OUTPUT_DIR}${NC}"
    echo ""
 #   
    # List generated files
    if [ -f "$OUTPUT_DIR/flops_profile.json" ]; then
        echo "Generated files:"
        ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
    fi
    
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "   1. Review FLOPS breakdown by component:"
    echo "      cat $OUTPUT_DIR/flops_profile.json | jq '.flops_analysis.evoformer_breakdown'"
    echo ""
    echo "   2. Check Model FLOPS Utilization (MFU):"
    echo "      cat $OUTPUT_DIR/flops_profile.json | jq '.efficiency_metrics'"
    echo ""
    
    if [ -f "$OUTPUT_DIR/computational_intensity.json" ]; then
        echo "   3. View computational intensity analysis:"
        echo "      cat $OUTPUT_DIR/computational_intensity.json"
        echo ""
    fi
    
    if [ -f "$OUTPUT_DIR/roofline_data.json" ]; then
        echo "   4. Review roofline model data:"
        echo "      cat $OUTPUT_DIR/roofline_data.json | jq '.optimization_targets'"
        echo ""
    fi
    
    echo "   5. Compare with PyTorch profiler results:"
    echo "      diff <(cat $OUTPUT_DIR/flops_profile.json | jq) <(cat profiles/performance_summary.json | jq)"
    
else
    echo -e "${RED}✗ FLOPS profiling failed with exit code $EXIT_CODE${NC}"
    exit $EXIT_CODE
fi

echo ""
echo "========================================================================"

