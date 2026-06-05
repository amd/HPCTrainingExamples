#!/bin/bash
#
# PyTorch Profiler Runner for Tiny OpenFold V2 (Fused)
#
# This script provides convenient wrapper for running PyTorch profiling
# with various fusion configurations and analysis options.
#
# Usage:
#   ./run_pytorch_profiler.sh                    # Default: all fusions enabled
#   ./run_pytorch_profiler.sh --baseline         # Disable all fusions (baseline)
#   ./run_pytorch_profiler.sh --ablation         # Run ablation study
#   ./run_pytorch_profiler.sh --compare-v1       # Compare with V1 baseline

set -e

# Default configuration
BATCH_SIZE=4
SEQ_LEN=64
NUM_BLOCKS=4
NUM_SEQS=16
NUM_STEPS=20
PROFILE_STEPS=5
WARMUP_STEPS=3
DEVICE=""
PROFILE_DIR="./pytorch_profiles_v2"
MODE="default"

# Fusion flags
DISABLE_QKV_MSA=""
DISABLE_QKV_TRIANGLE=""
DISABLE_FLASH=""
DISABLE_TRIANGLE=""
ENABLE_COMPILE=""
DISABLE_ALL=""

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
        --profile-steps)
            PROFILE_STEPS="$2"
            shift 2
            ;;
        --device)
            DEVICE="--device $2"
            shift 2
            ;;
        --profile-dir)
            PROFILE_DIR="$2"
            shift 2
            ;;
        --baseline)
            MODE="baseline"
            DISABLE_ALL="--disable-all-fusion"
            shift
            ;;
        --ablation)
            MODE="ablation"
            shift
            ;;
        --compare-v1)
            MODE="compare"
            shift
            ;;
        --disable-qkv-msa)
            DISABLE_QKV_MSA="--disable-qkv-fusion-msa"
            shift
            ;;
        --disable-qkv-triangle)
            DISABLE_QKV_TRIANGLE="--disable-qkv-fusion-triangle"
            shift
            ;;
        --disable-flash)
            DISABLE_FLASH="--disable-flash-attention"
            shift
            ;;
        --disable-triangle)
            DISABLE_TRIANGLE="--disable-triangle-fusion"
            shift
            ;;
        --enable-compile)
            ENABLE_COMPILE="--enable-torch-compile"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --batch-size N          Batch size (default: 4)"
            echo "  --seq-len N             Sequence length (default: 64)"
            echo "  --num-blocks N          Number of Evoformer blocks (default: 4)"
            echo "  --num-seqs N            Number of MSA sequences (default: 16)"
            echo "  --num-steps N           Total training steps (default: 20)"
            echo "  --profile-steps N       Steps to profile (default: 5)"
            echo "  --device N              GPU device ID"
            echo "  --profile-dir DIR       Profile output directory"
            echo ""
            echo "Modes:"
            echo "  --baseline              Disable all fusions (baseline comparison)"
            echo "  --ablation              Run ablation study (all fusion combinations)"
            echo "  --compare-v1            Compare with V1 baseline"
            echo ""
            echo "Fusion Control:"
            echo "  --disable-qkv-msa       Disable MSA QKV fusion"
            echo "  --disable-qkv-triangle  Disable triangle QKV fusion"
            echo "  --disable-flash         Disable Flash Attention"
            echo "  --disable-triangle      Disable triangle fusion"
            echo "  --enable-compile        Enable torch.compile"
            echo ""
            echo "Examples:"
            echo "  $0                                    # All fusions enabled"
            echo "  $0 --baseline                         # No fusions (baseline)"
            echo "  $0 --disable-flash --device 0         # All except Flash Attention"
            echo "  $0 --ablation                         # Run ablation study"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "======================================================================"
echo "Tiny OpenFold V2 - PyTorch Profiler"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Sequence length: $SEQ_LEN"
echo "  Evoformer blocks: $NUM_BLOCKS"
echo "  MSA sequences: $NUM_SEQS"
echo "  Profile steps: $PROFILE_STEPS / $NUM_STEPS"
echo "  Mode: $MODE"
echo "  Profile directory: $PROFILE_DIR"
echo ""

# Run based on mode
case $MODE in
    default)
        echo "Running profiling with all fusions enabled..."
        python run_pytorch_profiler.py \
            --batch-size $BATCH_SIZE \
            --seq-len $SEQ_LEN \
            --num-blocks $NUM_BLOCKS \
            --num-seqs $NUM_SEQS \
            --num-steps $NUM_STEPS \
            --profile-steps $PROFILE_STEPS \
            --warmup-steps $WARMUP_STEPS \
            --profile-dir $PROFILE_DIR \
            $DEVICE \
            $DISABLE_QKV_MSA \
            $DISABLE_QKV_TRIANGLE \
            $DISABLE_FLASH \
            $DISABLE_TRIANGLE \
            $ENABLE_COMPILE \
            $DISABLE_ALL \
            --generate-report
        ;;
        
    baseline)
        echo "Running baseline profiling (all fusions disabled)..."
        python run_pytorch_profiler.py \
            --batch-size $BATCH_SIZE \
            --seq-len $SEQ_LEN \
            --num-blocks $NUM_BLOCKS \
            --num-seqs $NUM_SEQS \
            --num-steps $NUM_STEPS \
            --profile-steps $PROFILE_STEPS \
            --warmup-steps $WARMUP_STEPS \
            --profile-dir "${PROFILE_DIR}_baseline" \
            $DEVICE \
            --disable-all-fusion \
            --generate-report
        ;;
        
    ablation)
        echo "Running ablation study..."
        echo "This will test all fusion combinations..."
        echo ""
        
        # Create ablation directory
        ABLATION_DIR="${PROFILE_DIR}_ablation_$(date +%Y%m%d_%H%M%S)"
        mkdir -p $ABLATION_DIR
        
        # Test configurations
        configs=(
            "all_disabled:--disable-all-fusion"
            "only_qkv_msa:--disable-qkv-fusion-triangle --disable-flash-attention --disable-triangle-fusion"
            "only_flash:--disable-qkv-fusion-msa --disable-qkv-fusion-triangle --disable-triangle-fusion"
            "only_triangle:--disable-qkv-fusion-msa --disable-qkv-fusion-triangle --disable-flash-attention"
            "all_enabled:"
        )
        
        for config in "${configs[@]}"; do
            name="${config%%:*}"
            flags="${config#*:}"
            
            echo "Testing configuration: $name"
            python run_pytorch_profiler.py \
                --batch-size $BATCH_SIZE \
                --seq-len $SEQ_LEN \
                --num-blocks $NUM_BLOCKS \
                --num-seqs $NUM_SEQS \
                --num-steps $NUM_STEPS \
                --profile-steps $PROFILE_STEPS \
                --warmup-steps $WARMUP_STEPS \
                --profile-dir "${ABLATION_DIR}/${name}" \
                $DEVICE \
                $flags \
                --generate-report
            
            echo ""
        done
        
        echo "Ablation study complete!"
        echo "Results saved to: $ABLATION_DIR"
        ;;
        
    compare)
        echo "Running comparison with V1 baseline..."
        
        V1_PROFILE="../version1_pytorch_baseline/pytorch_profiles"
        
        if [ ! -d "$V1_PROFILE" ]; then
            echo "Warning: V1 profile directory not found: $V1_PROFILE"
            echo "Running V1 profiling first..."
            
            # Run V1 profiling if not exists
            pushd ../version1_pytorch_baseline > /dev/null
            if [ -f "run_pytorch_profiler.sh" ]; then
                ./run_pytorch_profiler.sh --batch-size $BATCH_SIZE --seq-len $SEQ_LEN
            else
                echo "Error: V1 profiling script not found"
                exit 1
            fi
            popd > /dev/null
        fi
        
        # Run V2 profiling
        python run_pytorch_profiler.py \
            --batch-size $BATCH_SIZE \
            --seq-len $SEQ_LEN \
            --num-blocks $NUM_BLOCKS \
            --num-seqs $NUM_SEQS \
            --num-steps $NUM_STEPS \
            --profile-steps $PROFILE_STEPS \
            --warmup-steps $WARMUP_STEPS \
            --profile-dir $PROFILE_DIR \
            $DEVICE \
            --generate-report \
            --compare-with-v1 $V1_PROFILE
        
        echo ""
        echo "Comparison complete!"
        echo "V1 results: $V1_PROFILE"
        echo "V2 results: $PROFILE_DIR"
        ;;
esac

echo ""
echo "======================================================================"
echo "Profiling Complete!"
echo "======================================================================"
echo ""
echo "Results saved to: $PROFILE_DIR"
echo ""

# Extract and display throughput information from fusion_analysis.json
if [ -f "${PROFILE_DIR}/fusion_analysis.json" ]; then
    echo "======================================================================"
    echo "Performance Summary"
    echo "======================================================================"
    
    # Extract throughput stats using Python
    python3 << EOF 2>/dev/null || echo "  (Throughput information not available)"
import json
import sys

try:
    with open('${PROFILE_DIR}/fusion_analysis.json', 'r') as f:
        data = json.load(f)
    
    throughput = data.get('throughput_statistics', {})
    if throughput:
        print(f"  Total steps:           {throughput.get('total_steps', 'N/A')}")
        print(f"  Batch size:            {throughput.get('batch_size', 'N/A')}")
        print(f"  Total samples:         {throughput.get('total_samples', 'N/A')}")
        print(f"  Total time:            {throughput.get('total_time_sec', 0):.2f} seconds")
        print(f"  Average step time:     {throughput.get('avg_step_time_ms', 0):.2f} ms")
        print(f"  Average throughput:     {throughput.get('avg_throughput_samples_per_sec', 0):.2f} samples/sec")
        print(f"  Min step time:         {throughput.get('min_step_time_ms', 0):.2f} ms")
        print(f"  Max step time:         {throughput.get('max_step_time_ms', 0):.2f} ms")
    else:
        print("  (Throughput information not available)")
except Exception as e:
    print(f"  (Error reading throughput data: {e})")
EOF
    echo ""
fi

echo "To analyze results:"
echo "  1. View comprehensive report:"
echo "     less ${PROFILE_DIR}/comprehensive_profiling_report.md"
echo ""
echo "  2. View in Chrome (detailed timeline):"
echo "     Open chrome://tracing"
echo "     Load: ${PROFILE_DIR}/*.pt.trace.json"
echo ""
echo "  3. View in TensorBoard:"
echo "     tensorboard --logdir ${PROFILE_DIR}"
echo ""
echo "  4. View fusion analysis:"
echo "     cat ${PROFILE_DIR}/fusion_analysis.json | python -m json.tool"
echo ""


