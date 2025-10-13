#!/bin/bash
#
# AI Workshop - Version 1 Performance Study Launcher
#
# Runs PyTorch baseline with configurable problem sizes to demonstrate
# performance characteristics and establish baseline metrics.
#
# Usage: ./launch_performance_study.sh [tiny|medium|large|very_large] [--enable-profilers]
#

set -e

# Default to medium if no argument provided
SIZE=${1:-medium}
ENABLE_PROFILERS=false

# Check for profiler flag
if [[ "$2" == "--enable-profilers" ]] || [[ "$1" == "--enable-profilers" ]]; then
    ENABLE_PROFILERS=true
    if [[ "$1" == "--enable-profilers" ]]; then
        SIZE=${2:-medium}
    fi
fi

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
        echo "Usage: $0 [SIZE] [--enable-profilers]"
        echo ""
        echo "Available sizes:"
        echo "  tiny        - Quick validation    (~2M params,    <5s/iter)"
        echo "  small       - Development test    (~21M params,   10-30s/iter)"
        echo "  medium      - Workshop standard   (~167M params,  30-60s/iter)"
        echo "  large       - Serious benchmark   (~1.3B params,  1-3min/iter)"
        echo "  very_large  - Stress test         (~10.7B params, 5-10min/iter)"
        echo ""
        echo "Options:"
        echo "  --enable-profilers    Enable PyTorch and DeepSpeed profilers"
        echo ""
        echo "Examples:"
        echo "  $0 medium"
        echo "  $0 large --enable-profilers"
        exit 1
        ;;
esac

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="performance_results_${SIZE}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Display configuration
echo "================================================================================"
echo "CASTILLE AI WORKSHOP - VERSION 1 BASELINE PERFORMANCE STUDY"
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
echo "  Profilers Enabled:   $ENABLE_PROFILERS"
echo ""
echo "Output Directory: $OUTPUT_DIR"
echo "================================================================================"
echo ""

# Save configuration metadata
cat > "$OUTPUT_DIR/config.json" <<EOF
{
  "version": "v1_baseline",
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
  "profilers_enabled": $ENABLE_PROFILERS
}
EOF

# Build command
CMD="python tiny_llama_v1.py \
    --hidden-dim $HIDDEN_DIM \
    --num-layers $NUM_LAYERS \
    --seq-len $SEQ_LEN \
    --batch-size $BATCH_SIZE \
    --num-steps $NUM_STEPS \
    --profile-dir $OUTPUT_DIR"

# Add profiler flags if enabled
if [ "$ENABLE_PROFILERS" = true ]; then
    CMD="$CMD --enable-all-profiling"
    echo "Note: Profiling enabled - this will increase runtime"
    echo ""
fi

# Run training
echo "Starting V1 Baseline training..."
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

# Try to extract throughput from the log
if grep -q "Throughput:" "$OUTPUT_DIR/training_output.log"; then
    THROUGHPUT=$(grep "Throughput:" "$OUTPUT_DIR/training_output.log" | tail -1 | awk '{print $2, $3}')
    echo "Throughput: $THROUGHPUT"
fi

if grep -q "Peak memory usage:" "$OUTPUT_DIR/training_output.log"; then
    MEMORY=$(grep "Peak memory usage:" "$OUTPUT_DIR/training_output.log" | tail -1 | awk '{print $4, $5}')
    echo "Peak Memory: $MEMORY"
fi

echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo "  - config.json          Configuration metadata"
echo "  - training_output.log  Full training output"
if [ "$ENABLE_PROFILERS" = true ]; then
    echo "  - performance_summary.json  Performance metrics"
    echo "  - pytorch_profiles/    PyTorch profiler data (if enabled)"
fi

echo ""
echo "Next Steps:"
echo "  1. Review results: cat $OUTPUT_DIR/training_output.log"
if [ "$ENABLE_PROFILERS" = true ]; then
    echo "  2. Analyze profiling: tensorboard --logdir $OUTPUT_DIR"
fi
echo "  3. Run V2 for comparison: cd ../version2_pytorch_fused && ./launch_performance_study.sh $SIZE"
echo "  4. Compare all versions: python ../compare_versions.py"
echo ""
