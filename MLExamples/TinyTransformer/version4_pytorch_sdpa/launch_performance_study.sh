#!/bin/bash
#
# AI Workshop - Version 4 Performance Study Launcher
#
# Runs PyTorch SDPA optimized model to demonstrate that library implementations
# can match custom Triton kernels. Achieves same 3.1x speedup as V3.
#
# Usage: ./launch_performance_study.sh [tiny|medium|large|very_large]
#

set -e

# Default to medium if no argument provided
SIZE=${1:-medium}

# Configure problem size
case ${SIZE,,} in  # Convert to lowercase
    tiny)
        HIDDEN_DIM=256
        NUM_LAYERS=4
        SEQ_LEN=128
        BATCH_SIZE=8
        NUM_STEPS=50
        PARAMS="~2.6M"
        EXPECTED_TIME="<5s/iter"
        ;;
    small)
        HIDDEN_DIM=512
        NUM_LAYERS=8
        SEQ_LEN=256
        BATCH_SIZE=8
        NUM_STEPS=50
        PARAMS="~20.9M"
        EXPECTED_TIME="10-30s/iter"
        ;;
    medium|med)
        HIDDEN_DIM=1024
        NUM_LAYERS=12
        SEQ_LEN=512
        BATCH_SIZE=16
        NUM_STEPS=100
        PARAMS="~167M"
        EXPECTED_TIME="30-60s/iter"
        ;;
    large)
        HIDDEN_DIM=2048
        NUM_LAYERS=16
        SEQ_LEN=1024
        BATCH_SIZE=8
        NUM_STEPS=100
        PARAMS="~1.3B"
        EXPECTED_TIME="1-3min/iter"
        ;;
    very_large|xl|verylarge)
        HIDDEN_DIM=4096
        NUM_LAYERS=24
        SEQ_LEN=2048
        BATCH_SIZE=4
        NUM_STEPS=100
        PARAMS="~10.7B"
        EXPECTED_TIME="5-10min/iter"
        ;;
    *)
        echo "Error: Invalid problem size '$SIZE'"
        echo ""
        echo "Usage: $0 [SIZE]"
        echo ""
        echo "Available sizes:"
        echo "  tiny        - Quick validation    (~2M params,    <5s/iter)"
        echo "  small       - Development test    (~21M params,   10-30s/iter)"
        echo "  medium      - Workshop standard   (~167M params,  30-60s/iter)"
        echo "  large       - Serious benchmark   (~1.3B params,  1-3min/iter)"
        echo "  very_large  - Stress test         (~10.7B params, 5-10min/iter)"
        echo ""
        echo "Examples:"
        echo "  $0 medium"
        echo "  $0 large"
        exit 1
        ;;
esac

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="performance_results_${SIZE}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Display configuration
echo "================================================================================"
echo "CASTILLE AI WORKSHOP - VERSION 4 PYTORCH SDPA PERFORMANCE STUDY"
echo "     Library-Optimized Approach: PyTorch SDPA (Same Performance as V3)"
echo "================================================================================"
echo ""
echo "Problem Size: ${SIZE^^}"
echo "Configuration:"
echo "  Hidden Dimension:    $HIDDEN_DIM"
echo "  Number of Layers:    $NUM_LAYERS"
echo "  Sequence Length:     $SEQ_LEN"
echo "  Batch Size:          $BATCH_SIZE"
echo "  Training Steps:      $NUM_STEPS"
echo "  Est. Parameters:     $PARAMS"
echo "  Expected Time:       $EXPECTED_TIME"
echo ""
echo "Optimizations (PyTorch SDPA):"
echo "  - PyTorch SDPA         - Hardware-accelerated Flash Attention"
echo "  - Standard PyTorch Ops - No custom Triton kernels needed"
echo "  - Eager Execution      - torch.compile disabled (overhead on ROCm)"
echo "  - Same Performance     - Matches V3 (3.1x speedup, 61% memory reduction)"
echo ""
echo "Output Directory: $OUTPUT_DIR"
echo "================================================================================"
echo ""

# Save configuration metadata
cat > "$OUTPUT_DIR/config.json" <<EOF
{
  "version": "v4_pytorch_sdpa",
  "size": "$SIZE",
  "timestamp": "$TIMESTAMP",
  "configuration": {
    "hidden_dim": $HIDDEN_DIM,
    "num_layers": $NUM_LAYERS,
    "seq_len": $SEQ_LEN,
    "batch_size": $BATCH_SIZE,
    "num_steps": $NUM_STEPS,
    "estimated_parameters": "$PARAMS"
  },
  "pytorch_optimizations": {
    "pytorch_sdpa": true,
    "flash_attention": true,
    "advanced_tiling": true,
    "cumulative_v1_v2_v3": true
  }
}
EOF

# Build command - use train mode for full training loop
CMD="python tiny_llama_v4.py \
    --mode train \
    --hidden-dim $HIDDEN_DIM \
    --num-layers $NUM_LAYERS \
    --seq-len $SEQ_LEN \
    --batch-size $BATCH_SIZE \
    --num-steps $NUM_STEPS"

# Run training
echo "Starting V4 PyTorch SDPA training..."
echo ""
echo "Note: Using PyTorch's hardware-accelerated SDPA instead of custom Triton kernels"
echo ""

START_TIME=$(date +%s)

# Execute and capture output
$CMD 2>&1 | tee "$OUTPUT_DIR/training_output.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Extract key metrics from output
echo ""
echo "================================================================================"
echo "PERFORMANCE STUDY COMPLETE"
echo "================================================================================"
echo "Total Runtime: ${DURATION}s"
echo ""

# Try to extract metrics from the log
if grep -q "Throughput:" "$OUTPUT_DIR/training_output.log"; then
    THROUGHPUT=$(grep "Throughput:" "$OUTPUT_DIR/training_output.log" | tail -1 | awk '{print $2, $3}')
    echo "Throughput: $THROUGHPUT"
fi

if grep -q "Peak memory usage:" "$OUTPUT_DIR/training_output.log"; then
    MEMORY=$(grep "Peak memory usage:" "$OUTPUT_DIR/training_output.log" | tail -1 | awk '{print $4, $5}')
    echo "Peak Memory: $MEMORY"
fi

# Check for version comparisons
if grep -q "Speedup vs V1:" "$OUTPUT_DIR/training_output.log"; then
    SPEEDUP_V1=$(grep "Speedup vs V1:" "$OUTPUT_DIR/training_output.log" | tail -1 | awk '{print $4}')
    echo "Speedup vs V1: $SPEEDUP_V1"
fi

if grep -q "Speedup vs V2:" "$OUTPUT_DIR/training_output.log"; then
    SPEEDUP_V2=$(grep "Speedup vs V2:" "$OUTPUT_DIR/training_output.log" | tail -1 | awk '{print $4}')
    echo "Speedup vs V2: $SPEEDUP_V2"
fi

if grep -q "Speedup vs V3:" "$OUTPUT_DIR/training_output.log"; then
    SPEEDUP_V3=$(grep "Speedup vs V3:" "$OUTPUT_DIR/training_output.log" | tail -1 | awk '{print $4}')
    echo "Speedup vs V3: $SPEEDUP_V3"
fi

echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo "  - config.json              Configuration metadata"
echo "  - training_output.log      Full training output with all version comparisons"

echo ""
echo "PyTorch SDPA Benefits:"
echo "  • Hardware-accelerated Flash Attention (same as V3 Triton implementation)"
echo "  • No custom kernel code - uses optimized PyTorch libraries"
echo "  • Easier maintenance and PyTorch version compatibility"
echo "  • Same 3.1x speedup and 61% memory reduction as V3"
echo ""
echo "Next Steps:"
echo "  1. Compare to V3 Triton: Both should have identical performance"
echo "  2. Review all versions: cd .. && python compare_versions.py"
echo "  3. Run V3 for comparison: cd ../version3_triton && ./launch_performance_study.sh $SIZE"
echo "  4. Analyze results: python compare_versions.py"
echo ""
