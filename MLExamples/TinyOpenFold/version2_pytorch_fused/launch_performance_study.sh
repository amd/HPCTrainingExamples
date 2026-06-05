#!/bin/bash

# Performance Study Launcher for Tiny OpenFold V2
# Automates comparative performance analysis across configurations

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Default configuration
STUDY_NAME="performance_study_$(date +%Y%m%d_%H%M%S)"
NUM_RUNS=3
BATCH_SIZES="2 4 8"
SEQ_LENS="32 64 128"
NUM_STEPS=50
DEVICE=0
RUN_BASELINE=true
RUN_ABLATION=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --study-name) STUDY_NAME="$2"; shift 2 ;;
        --num-runs) NUM_RUNS="$2"; shift 2 ;;
        --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
        --seq-lens) SEQ_LENS="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --no-baseline) RUN_BASELINE=false; shift ;;
        --ablation) RUN_ABLATION=true; shift ;;
        --help|-h)
            echo "Performance Study Launcher for Tiny OpenFold V2"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --study-name NAME       Study name (default: timestamped)"
            echo "  --num-runs N            Number of runs per config (default: 3)"
            echo "  --batch-sizes \"N...\"    Batch sizes to test (default: \"2 4 8\")"
            echo "  --seq-lens \"N...\"       Sequence lengths to test (default: \"32 64 128\")"
            echo "  --num-steps N           Training steps per run (default: 50)"
            echo "  --device N              GPU device (default: 0)"
            echo "  --no-baseline           Skip baseline comparison"
            echo "  --ablation              Run fusion ablation study"
            echo ""
            echo "Examples:"
            echo "  $0                                          # Standard study"
            echo "  $0 --num-runs 5 --batch-sizes \"4 8 16\"    # Custom config"
            echo "  $0 --ablation                               # With ablation study"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$STUDY_NAME"
cd "$STUDY_NAME"

log_info "======================================================================"
log_info "Tiny OpenFold V2 - Performance Study"
log_info "======================================================================"
echo ""
log_info "Study Configuration:"
log_info "  Study name: $STUDY_NAME"
log_info "  Runs per configuration: $NUM_RUNS"
log_info "  Batch sizes: $BATCH_SIZES"
log_info "  Sequence lengths: $SEQ_LENS"
log_info "  Steps per run: $NUM_STEPS"
log_info "  Device: $DEVICE"
log_info "  Run baseline: $RUN_BASELINE"
log_info "  Run ablation: $RUN_ABLATION"
echo ""

# Save configuration
cat > config.json << EOF
{
    "study_name": "$STUDY_NAME",
    "num_runs": $NUM_RUNS,
    "batch_sizes": [$BATCH_SIZES],
    "seq_lens": [$SEQ_LENS],
    "num_steps": $NUM_STEPS,
    "device": $DEVICE,
    "run_baseline": $RUN_BASELINE,
    "run_ablation": $RUN_ABLATION,
    "timestamp": "$(date --iso-8601=seconds)"
}
EOF

# Main study: All fusions enabled
log_step "Running main performance study (all fusions enabled)..."

for batch_size in $BATCH_SIZES; do
    for seq_len in $SEQ_LENS; do
        config_name="b${batch_size}_s${seq_len}"
        log_info "Testing configuration: batch_size=$batch_size, seq_len=$seq_len"
        
        for run in $(seq 1 $NUM_RUNS); do
            log_info "  Run $run/$NUM_RUNS..."
            python ../tiny_openfold_v2.py \
                --batch-size $batch_size \
                --seq-len $seq_len \
                --num-steps $NUM_STEPS \
                --profile-dir "${config_name}_run${run}" \
                > "${config_name}_run${run}.log" 2>&1
        done
        
        log_info "  ✓ Configuration complete"
    done
done

# Baseline comparison
if [ "$RUN_BASELINE" = true ]; then
    log_step "Running baseline comparison (all fusions disabled)..."
    
    for batch_size in $BATCH_SIZES; do
        for seq_len in $SEQ_LENS; do
            config_name="b${batch_size}_s${seq_len}_baseline"
            log_info "Testing baseline: batch_size=$batch_size, seq_len=$seq_len"
            
            for run in $(seq 1 $NUM_RUNS); do
                log_info "  Run $run/$NUM_RUNS..."
                python ../tiny_openfold_v2.py \
                    --batch-size $batch_size \
                    --seq-len $seq_len \
                    --num-steps $NUM_STEPS \
                    --disable-all-fusion \
                    --profile-dir "${config_name}_run${run}" \
                    > "${config_name}_run${run}.log" 2>&1
            done
            
            log_info "  ✓ Baseline complete"
        done
    done
fi

# Ablation study
if [ "$RUN_ABLATION" = true ]; then
    log_step "Running fusion ablation study..."
    
    # Use middle configuration
    BATCH_SIZE=$(echo $BATCH_SIZES | awk '{print $2}')
    SEQ_LEN=$(echo $SEQ_LENS | awk '{print $2}')
    [ -z "$BATCH_SIZE" ] && BATCH_SIZE=$(echo $BATCH_SIZES | awk '{print $1}')
    [ -z "$SEQ_LEN" ] && SEQ_LEN=$(echo $SEQ_LENS | awk '{print $1}')
    
    log_info "Using batch_size=$BATCH_SIZE, seq_len=$SEQ_LEN for ablation"
    
    # Test each fusion individually
    ABLATIONS=(
        "all_disabled:--disable-all-fusion"
        "only_qkv_msa:--disable-qkv-fusion-triangle --disable-flash-attention --disable-triangle-fusion"
        "only_qkv_triangle:--disable-qkv-fusion-msa --disable-flash-attention --disable-triangle-fusion"
        "only_flash:--disable-qkv-fusion-msa --disable-qkv-fusion-triangle --disable-triangle-fusion"
        "only_triangle:--disable-qkv-fusion-msa --disable-qkv-fusion-triangle --disable-flash-attention"
        "no_qkv:--disable-qkv-fusion-msa --disable-qkv-fusion-triangle"
        "no_flash:--disable-flash-attention"
        "no_triangle:--disable-triangle-fusion"
        "all_enabled:"
    )
    
    for ablation in "${ABLATIONS[@]}"; do
        name="${ablation%%:*}"
        flags="${ablation#*:}"
        
        log_info "Testing ablation: $name"
        
        for run in $(seq 1 $NUM_RUNS); do
            python ../tiny_openfold_v2.py \
                --batch-size $BATCH_SIZE \
                --seq-len $SEQ_LEN \
                --num-steps $NUM_STEPS \
                $flags \
                --profile-dir "ablation_${name}_run${run}" \
                > "ablation_${name}_run${run}.log" 2>&1
        done
        
        log_info "  ✓ Ablation $name complete"
    done
fi

# Analyze results
log_step "Analyzing results..."

python3 << 'ANALYSIS_SCRIPT'
import json
import glob
import re
import numpy as np
from pathlib import Path

results = []

# Parse all performance summary files
for json_file in glob.glob("*/performance_summary_v2.json"):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        config = data.get('config', {})
        perf = data.get('performance_summary', {})
        fusion = data.get('fusion_statistics', {})
        
        # Extract configuration from path
        path_parts = Path(json_file).parts[0]
        
        results.append({
            'config': path_parts,
            'batch_size': config.get('max_seq_len', 'N/A'),
            'seq_len': config.get('max_seq_len', 'N/A'),
            'speed': perf.get('avg_training_speed', 0),
            'memory_mb': perf.get('peak_memory_mb', 0),
            'batch_time_ms': perf.get('avg_batch_time', 0) * 1000,
            'loss': perf.get('avg_loss', 0),
            'fusion_enabled': fusion.get('qkv_fusion_msa_enabled', False)
        })
    except Exception as e:
        print(f"Error parsing {json_file}: {e}")

# Group by configuration
configs = {}
for result in results:
    config = result['config']
    if config not in configs:
        configs[config] = []
    configs[config].append(result)

# Generate summary
print("\n" + "="*80)
print("PERFORMANCE STUDY SUMMARY")
print("="*80)

for config_name in sorted(configs.keys()):
    runs = configs[config_name]
    speeds = [r['speed'] for r in runs if r['speed'] > 0]
    memories = [r['memory_mb'] for r in runs if r['memory_mb'] > 0]
    batch_times = [r['batch_time_ms'] for r in runs if r['batch_time_ms'] > 0]
    
    if speeds:
        print(f"\nConfiguration: {config_name}")
        print(f"  Runs: {len(runs)}")
        print(f"  Speed: {np.mean(speeds):.2f} ± {np.std(speeds):.2f} samples/sec")
        print(f"  Memory: {np.mean(memories):.1f} ± {np.std(memories):.1f} MB")
        print(f"  Batch time: {np.mean(batch_times):.2f} ± {np.std(batch_times):.2f} ms")

print("\n" + "="*80)

# Save results
with open('results_summary.json', 'w') as f:
    json.dump(configs, f, indent=2)

print("\nDetailed results saved to: results_summary.json")

ANALYSIS_SCRIPT

cd - > /dev/null

log_info "======================================================================"
log_info "Performance Study Complete!"
log_info "======================================================================"
echo ""
log_info "Study directory: $STUDY_NAME"
echo ""
log_info "Generated files:"
log_info "  - config.json                 : Study configuration"
log_info "  - results_summary.json        : Aggregated results"
log_info "  - *.log                       : Individual run logs"
log_info "  - */performance_summary_v2.json : Detailed per-run data"
echo ""
log_info "To visualize results:"
log_info "  python ../analyze_performance_study.py --study-dir $STUDY_NAME"
echo ""


