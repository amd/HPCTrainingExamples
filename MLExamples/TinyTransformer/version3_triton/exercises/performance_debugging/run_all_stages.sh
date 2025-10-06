#!/bin/bash

#===============================================================================
# Performance Debugging Exercise - Run All Stages
#===============================================================================
# This script runs all V3 optimization stages and collects performance data
# for comparison and analysis. It demonstrates the systematic optimization
# process from broken implementation to high-performance optimized version.
#===============================================================================

set -e  # Exit on error

EXERCISE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V3_DIR="$(dirname "$(dirname "$EXERCISE_DIR")")"
RESULTS_DIR="$EXERCISE_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create results directory
mkdir -p "$RESULTS_DIR"

# Configuration
BATCH_SIZE=8
SEQ_LEN=128
NUM_STEPS=20
PROFILE_STEPS=10  # Shorter runs for profiling to keep trace files manageable

echo "================================================================================"
echo "V3 PERFORMANCE DEBUGGING EXERCISE - ALL STAGES"
echo "================================================================================"
echo "This script runs the complete optimization journey for V3 Triton kernels"
echo ""
echo "Configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Sequence length: $SEQ_LEN"
echo "  Training steps: $NUM_STEPS (full run)"
echo "  Profile steps: $PROFILE_STEPS (for rocprof)"
echo "  Results directory: $RESULTS_DIR"
echo "================================================================================"
echo ""

# Function to run a stage
run_stage() {
    local stage_num=$1
    local stage_name=$2
    local stage_file=$3
    local description=$4

    echo ""
    echo "================================================================================"
    echo "STAGE $stage_num: $stage_name"
    echo "================================================================================"
    echo "$description"
    echo ""

    # Run without profiling for full metrics
    echo "Running stage $stage_num (full run: $NUM_STEPS steps)..."
    python "$stage_file" --batch-size $BATCH_SIZE --seq-len $SEQ_LEN --num-steps $NUM_STEPS \
        2>&1 | tee "$RESULTS_DIR/stage${stage_num}_output.log"

    echo ""
    echo "Stage $stage_num complete. Output saved to: $RESULTS_DIR/stage${stage_num}_output.log"

    # Run with rocprof for detailed profiling (shorter run)
    if command -v rocprof &> /dev/null; then
        echo ""
        echo "Running stage $stage_num with rocprof ($PROFILE_STEPS steps for trace analysis)..."
        rocprof --stats \
            -o "$RESULTS_DIR/stage${stage_num}_profile.csv" \
            python "$stage_file" --batch-size $BATCH_SIZE --seq-len $SEQ_LEN --num-steps $PROFILE_STEPS \
            2>&1 | tee "$RESULTS_DIR/stage${stage_num}_profile.log"

        echo "Profile data saved to: $RESULTS_DIR/stage${stage_num}_profile.csv"
    else
        echo "rocprof not found - skipping detailed profiling for stage $stage_num"
        echo "To enable profiling, ensure ROCm tools are installed and rocprof is in PATH"
    fi

    echo ""
    echo "--------------------------------------------------------------------------------"
    sleep 2  # Brief pause between stages
}

# Stage 1: Broken Loss (Missing Weight Initialization)
run_stage 1 \
    "Broken Loss" \
    "$EXERCISE_DIR/v3_stage1_broken_loss.py" \
    "Problem: Loss = 942 instead of ~7
Root Cause: Missing weight initialization causes exploding logits
Expected: Loss ~942, training will diverge"

# Stage 2: Fixed Loss, Terrible Performance
run_stage 2 \
    "Slow Performance" \
    "$EXERCISE_DIR/v3_stage2_slow_performance.py" \
    "Problem: Loss fixed (7.0) but only 15.2 samples/sec (vs V1's 97 samples/sec)
Root Cause: Non-contiguous tensors after repeat_interleave for GQA
Expected: Loss ~7.0, Speed ~15 samples/sec, Time ~526ms per batch"

# Stage 3: Better Performance, Wrong Timing
run_stage 3 \
    "Fake Timing" \
    "$EXERCISE_DIR/v3_stage3_fake_timing.py" \
    "Problem: Improved to 310 samples/sec but timing breakdown is wrong
Root Cause: Missing CUDA synchronization for individual operation timing
Expected: Loss ~7.0, Speed ~310 samples/sec, but timing doesn't add up"

# Stage 4: Accurate Timing, Slow Kernels
run_stage 4 \
    "Slow Kernels" \
    "$EXERCISE_DIR/v3_stage4_slow_kernels.py" \
    "Problem: Forward pass is 25.5ms (2.4x slower than V1's 10.8ms)
Root Cause: Inefficient Triton SwiGLU kernel doing manual matrix multiplication
Expected: Loss ~7.0, Speed ~306 samples/sec, Forward time ~25ms"

# Stage 5: Final Optimized Version
echo ""
echo "================================================================================"
echo "STAGE 5: Final Optimized"
echo "================================================================================"
echo "Solution: Use PyTorch for matrix multiplies, Triton only for element-wise fusion
Result: 2065 samples/sec (5.5x faster than V1!)
Expected: Loss ~7.0, Speed ~2065 samples/sec, Time ~3.9ms"
echo ""

echo "Running stage 5 (full run: $NUM_STEPS steps)..."
cd "$V3_DIR"
python tiny_llama_v3.py --batch-size $BATCH_SIZE --seq-len $SEQ_LEN --num-steps $NUM_STEPS \
    2>&1 | tee "$RESULTS_DIR/stage5_output.log"

echo ""
echo "Stage 5 complete. Output saved to: $RESULTS_DIR/stage5_output.log"

if command -v rocprof &> /dev/null; then
    echo ""
    echo "Running stage 5 with rocprof ($PROFILE_STEPS steps for trace analysis)..."
    rocprof --stats \
        -o "$RESULTS_DIR/stage5_profile.csv" \
        python tiny_llama_v3.py --batch-size $BATCH_SIZE --seq-len $SEQ_LEN --num-steps $PROFILE_STEPS \
        2>&1 | tee "$RESULTS_DIR/stage5_profile.log"

    echo "Profile data saved to: $RESULTS_DIR/stage5_profile.csv"
fi

cd "$EXERCISE_DIR"

# Generate comparison summary
echo ""
echo "================================================================================"
echo "GENERATING PERFORMANCE COMPARISON"
echo "================================================================================"

python3 - <<'EOF'
import re
import sys
from pathlib import Path

results_dir = Path("results")

def extract_metrics(log_file):
    """Extract key metrics from stage output log."""
    try:
        with open(log_file) as f:
            content = f.read()

        # Extract final loss
        loss_match = re.search(r'Final loss:\s+([\d.]+)', content)
        loss = float(loss_match.group(1)) if loss_match else None

        # Extract average training speed
        speed_match = re.search(r'Average training speed:\s+([\d.]+)\s+samples/sec', content)
        speed = float(speed_match.group(1)) if speed_match else None

        # Extract average batch time
        batch_time_match = re.search(r'Average batch time:\s+([\d.]+)\s+ms', content)
        batch_time = float(batch_time_match.group(1)) if batch_time_match else None

        # Extract average forward time
        fwd_time_match = re.search(r'Average forward time:\s+([\d.]+)\s+ms', content)
        fwd_time = float(fwd_time_match.group(1)) if fwd_time_match else None

        # Extract peak memory
        mem_match = re.search(r'Peak memory usage:\s+([\d.]+)\s+MB', content)
        memory = float(mem_match.group(1)) if mem_match else None

        return {
            'loss': loss,
            'speed': speed,
            'batch_time': batch_time,
            'forward_time': fwd_time,
            'memory': memory
        }
    except Exception as e:
        print(f"Error extracting from {log_file}: {e}")
        return {}

# Extract metrics from each stage
stages = {
    'Stage 1: Broken Loss': 'stage1_output.log',
    'Stage 2: Slow Performance': 'stage2_output.log',
    'Stage 3: Fake Timing': 'stage3_output.log',
    'Stage 4: Slow Kernels': 'stage4_output.log',
    'Stage 5: Optimized': 'stage5_output.log'
}

print("\n" + "=" * 100)
print("PERFORMANCE COMPARISON SUMMARY")
print("=" * 100)
print(f"{'Stage':<30} {'Loss':<10} {'Speed':<15} {'Batch Time':<15} {'Fwd Time':<15} {'Memory':<12}")
print("-" * 100)

baseline_speed = None
for stage_name, log_file in stages.items():
    log_path = results_dir / log_file
    if log_path.exists():
        metrics = extract_metrics(log_path)
        loss = f"{metrics.get('loss', 0):.4f}" if metrics.get('loss') else 'N/A'
        speed = metrics.get('speed', 0)
        speed_str = f"{speed:.1f} samp/s" if speed else 'N/A'

        # Calculate speedup vs Stage 2 (first working version)
        if stage_name == 'Stage 2: Slow Performance' and speed:
            baseline_speed = speed
        if baseline_speed and speed:
            speedup = speed / baseline_speed
            speed_str = f"{speed:.1f} ({speedup:.2f}x)"

        batch = f"{metrics.get('batch_time', 0):.1f} ms" if metrics.get('batch_time') else 'N/A'
        fwd = f"{metrics.get('forward_time', 0):.1f} ms" if metrics.get('forward_time') else 'N/A'
        mem = f"{metrics.get('memory', 0):.1f} MB" if metrics.get('memory') else 'N/A'

        print(f"{stage_name:<30} {loss:<10} {speed_str:<15} {batch:<15} {fwd:<15} {mem:<12}")
    else:
        print(f"{stage_name:<30} {'LOG NOT FOUND':<70}")

print("=" * 100)
print("\nKey Observations:")
print("  • Stage 1: Broken loss due to missing weight initialization")
print("  • Stage 2: Correct loss but 6.4x slower than baseline (non-contiguous tensors)")
print("  • Stage 3: Better performance (20x faster) after adding .contiguous()")
print("  • Stage 4: Same speed as Stage 3 (timing was already accurate)")
print("  • Stage 5: OPTIMAL - 136x faster than Stage 2, 5.5x faster than V1 baseline!")
print("\nMemory Efficiency:")
print("  • V3 uses ~282 MB vs V1's 522 MB (46% reduction)")
print("  • Flash Attention avoids materializing full attention matrix")
print("\nNext Steps:")
print("  1. Analyze profile CSV files in results/ directory")
print("  2. Upload *_profile.csv to Excel/Google Sheets for visualization")
print("  3. For detailed kernel traces, run: rocprofv2 --kernel-trace ...")
print("  4. Compare kernel execution times between stages")
print("=" * 100)
EOF

echo ""
echo "================================================================================"
echo "ALL STAGES COMPLETE!"
echo "================================================================================"
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Files generated:"
echo "  • stage*_output.log - Full training output with metrics"
echo "  • stage*_profile.csv - rocprof profiling data (if available)"
echo "  • stage*_profile.log - Profiling run output"
echo ""
echo "Analysis Commands:"
echo "  # View stage outputs"
echo "  cat $RESULTS_DIR/stage*_output.log"
echo ""
echo "  # Compare performance summaries"
echo "  grep 'Performance Summary' $RESULTS_DIR/stage*_output.log"
echo ""
echo "  # View top kernels from profiling"
echo "  head -20 $RESULTS_DIR/stage*_profile.csv"
echo ""
echo "  # Generate detailed comparison"
echo "  python compare_stages.py"
echo ""
echo "================================================================================"
echo "Exercise complete! Review the README.md for detailed analysis guidance."
echo "================================================================================"
