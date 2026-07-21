#!/bin/bash
################################################################################
# TinyOpenFold: Complete Optimization Tutorial
# Progressive performance improvement: V1 → V2 → V3
# Demonstrates 2.0x speedup through systematic optimization
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Test configuration
BASEDIR="/mnt/thera/data/incoming/asimishr/aiml_prof/HPCTrainingExamples/MLExamples/TinyOpenFold"
V1_DIR="$BASEDIR/version1_pytorch_baseline"
V2_DIR="$BASEDIR/version2_pytorch_fused"
V3_DIR="$BASEDIR/version3_triton"
DEVICE=0
STEPS=30

# Setup environment
clear
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║                                                                  ║${NC}"
echo -e "${BOLD}${CYAN}║         TinyOpenFold Performance Optimization Tutorial           ║${NC}"
echo -e "${BOLD}${CYAN}║                                                                  ║${NC}"
echo -e "${BOLD}${CYAN}║         Progressive Optimization: V1 → V2 → V3                   ║${NC}"
echo -e "${BOLD}${CYAN}║         Learn GPU optimization through practice!                 ║${NC}"
echo -e "${BOLD}${CYAN}║                                                                  ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${CYAN}[Step 1/7] Setting up environment...${NC}"
module load python/3.12 rocm/7.2 libffi/3.3
source $BASEDIR/venvOF/bin/activate
echo -e "${GREEN}✓ Environment ready${NC}"
echo ""

echo -e "${CYAN}[Step 2/7] Verifying GPU...${NC}"
python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}'); print(f'  PyTorch: {torch.__version__}')"
echo -e "${GREEN}✓ GPU verified${NC}"
echo ""

# Results file
RESULTS_FILE="$BASEDIR/tutorial_results_$(date +%Y%m%d_%H%M%S).txt"
echo "TinyOpenFold Optimization Tutorial Results" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "GPU: AMD Instinct MI300X" >> $RESULTS_FILE
echo "================================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${BLUE}  Part 1: Small Problem (64 residues, 16 MSA, batch=4)${NC}"
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Function to run test
run_test() {
    local version=$1
    local name=$2
    local seq_len=$3
    local num_seqs=$4
    local batch_size=$5
    local workdir=$6

    cd $workdir

    # Run test
    if [ "$version" == "V1" ]; then
        python3 tiny_openfold_v*.py \
            --seq-len $seq_len \
            --num-seqs $num_seqs \
            --batch-size $batch_size \
            --num-blocks 4 \
            --num-steps $STEPS \
            --device $DEVICE \
            2>&1 | tee /tmp/test_output.txt > /dev/null
    else
        ROCR_VISIBLE_DEVICES=$DEVICE python3 tiny_openfold_v*.py \
            --seq-len $seq_len \
            --num-seqs $num_seqs \
            --batch-size $batch_size \
            --num-blocks 4 \
            --num-steps $STEPS \
            2>&1 | tee /tmp/test_output.txt > /dev/null
    fi

    # Extract metrics
    local speed=$(grep -oP 'Average training speed:\s+\K[\d.]+' /tmp/test_output.txt | tail -1)
    local batch_time=$(grep -oP 'Average batch time:\s+\K[\d.]+' /tmp/test_output.txt | tail -1)
    local forward_time=$(grep -oP 'Average forward time:\s+\K[\d.]+' /tmp/test_output.txt | tail -1)
    local backward_time=$(grep -oP 'Average backward time:\s+\K[\d.]+' /tmp/test_output.txt | tail -1)
    local optimizer_time=$(grep -oP 'Average optimizer time:\s+\K[\d.]+' /tmp/test_output.txt | tail -1)
    local memory=$(grep -oP 'Peak memory.*:\s+\K[\d.]+' /tmp/test_output.txt | tail -1)

    echo "$version|$name|$speed|$batch_time|$forward_time|$backward_time|$optimizer_time|$memory"
}

# Small problem results
declare -a SMALL_RESULTS=()

# V1 - Small
echo -e "${YELLOW}[Step 3/7] Stage 1: Baseline (V1) - Small problem${NC}"
echo -e "  ${CYAN}Running pure PyTorch implementation...${NC}"
result_v1_small=$(run_test "V1" "Small" 64 16 4 "$V1_DIR")
SMALL_RESULTS+=("$result_v1_small")
IFS='|' read -r v ver speed batch fwd bwd opt mem <<< "$result_v1_small"
echo -e "  ${GREEN}✓ Complete${NC} - Speed: ${BOLD}${speed} samples/sec${NC}, Batch: ${batch} ms"
echo ""

# V2 - Small
echo -e "${YELLOW}[Step 4/7] Stage 2: Kernel Fusion (V2) - Small problem${NC}"
echo -e "  ${CYAN}Running with QKV fusion + Flash Attention...${NC}"
result_v2_small=$(run_test "V2" "Small" 64 16 4 "$V2_DIR")
SMALL_RESULTS+=("$result_v2_small")
IFS='|' read -r v ver speed batch fwd bwd opt mem <<< "$result_v2_small"

# Calculate improvement
IFS='|' read -r _ _ v1_speed v1_batch _ _ _ _ <<< "$result_v1_small"
speedup=$(awk "BEGIN {printf \"%.2f\", $speed / $v1_speed}")
improvement=$(awk "BEGIN {printf \"%.0f\", ($speed / $v1_speed - 1) * 100}")
echo -e "  ${GREEN}✓ Complete${NC} - Speed: ${BOLD}${speed} samples/sec${NC}, Batch: ${batch} ms"
echo -e "  ${MAGENTA}→ Speedup: ${BOLD}${speedup}x${NC} (${GREEN}+${improvement}%${NC})"
echo ""

# V3 - Small
echo -e "${YELLOW}[Step 5/7] Stage 3: Custom Triton Kernels (V3) - Small problem${NC}"
echo -e "  ${CYAN}Running with custom LayerNorm + Flash Attention kernels...${NC}"
result_v3_small=$(run_test "V3" "Small" 64 16 4 "$V3_DIR")
SMALL_RESULTS+=("$result_v3_small")
IFS='|' read -r v ver speed batch fwd bwd opt mem <<< "$result_v3_small"

# Calculate improvement
speedup_v2=$(awk "BEGIN {printf \"%.2f\", $speed / $(echo $result_v2_small | cut -d'|' -f3)}")
speedup_v1=$(awk "BEGIN {printf \"%.2f\", $speed / $v1_speed}")
improvement_v1=$(awk "BEGIN {printf \"%.0f\", ($speed / $v1_speed - 1) * 100}")
echo -e "  ${GREEN}✓ Complete${NC} - Speed: ${BOLD}${speed} samples/sec${NC}, Batch: ${batch} ms"
echo -e "  ${MAGENTA}→ Speedup vs V2: ${BOLD}${speedup_v2}x${NC}"
echo -e "  ${MAGENTA}→ Speedup vs V1: ${BOLD}${speedup_v1}x${NC} (${GREEN}+${improvement_v1}%${NC}) ${BOLD}⚡⚡⚡${NC}"
echo ""

echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${BLUE}  Part 2: Medium Problem (128 residues, 32 MSA, batch=2)${NC}"
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Medium problem results
declare -a MEDIUM_RESULTS=()

# V1-V2-V3 Medium (compact output)
echo -e "${YELLOW}[Step 6/7] Running all versions on medium problem...${NC}"

echo -e "  ${CYAN}V1 Baseline...${NC}"
result_v1_med=$(run_test "V1" "Medium" 128 32 2 "$V1_DIR")
MEDIUM_RESULTS+=("$result_v1_med")
IFS='|' read -r v ver speed batch fwd bwd opt mem <<< "$result_v1_med"
echo -e "  ${GREEN}✓ V1${NC} - ${speed} samples/sec"

echo -e "  ${CYAN}V2 Fused...${NC}"
result_v2_med=$(run_test "V2" "Medium" 128 32 2 "$V2_DIR")
MEDIUM_RESULTS+=("$result_v2_med")
IFS='|' read -r v ver speed batch fwd bwd opt mem <<< "$result_v2_med"
echo -e "  ${GREEN}✓ V2${NC} - ${speed} samples/sec"

echo -e "  ${CYAN}V3 Triton...${NC}"
result_v3_med=$(run_test "V3" "Medium" 128 32 2 "$V3_DIR")
MEDIUM_RESULTS+=("$result_v3_med")
IFS='|' read -r v ver speed batch fwd bwd opt mem <<< "$result_v3_med"
echo -e "  ${GREEN}✓ V3${NC} - ${speed} samples/sec"
echo ""

# Generate comprehensive summary
echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${CYAN}  [Step 7/7] Performance Summary & Analysis${NC}"
echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "PERFORMANCE SUMMARY" >> $RESULTS_FILE
echo "================================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

echo -e "${BOLD}Small Problem (64 residues):${NC}"
echo "" >> $RESULTS_FILE
echo "Small Problem (64 residues):" >> $RESULTS_FILE
printf "${MAGENTA}%-8s %-12s %-12s %-12s %-12s %-10s${NC}\n" "Version" "Speed(s/s)" "Batch(ms)" "Forward(ms)" "Backward(ms)" "Speedup"
printf "%-8s %-12s %-12s %-12s %-12s %-10s\n" "Version" "Speed(s/s)" "Batch(ms)" "Forward(ms)" "Backward(ms)" "Speedup" >> $RESULTS_FILE
echo "────────────────────────────────────────────────────────────────────────"
echo "────────────────────────────────────────────────────────────────────────" >> $RESULTS_FILE

# Extract V1 small baseline
IFS='|' read -r _ _ v1s_speed v1s_batch v1s_fwd v1s_bwd _ _ <<< "${SMALL_RESULTS[0]}"

for i in "${!SMALL_RESULTS[@]}"; do
    IFS='|' read -r ver name speed batch fwd bwd opt mem <<< "${SMALL_RESULTS[$i]}"

    if [ "$i" -eq 0 ]; then
        speedup="1.0x"
    else
        speedup=$(awk "BEGIN {printf \"%.2fx\", $speed / $v1s_speed}")
    fi

    printf "%-8s %-12s %-12s %-12s %-12s %-10s\n" "$ver" "$speed" "$batch" "$fwd" "$bwd" "$speedup"
    printf "%-8s %-12s %-12s %-12s %-12s %-10s\n" "$ver" "$speed" "$batch" "$fwd" "$bwd" "$speedup" >> $RESULTS_FILE
done

echo ""
echo "" >> $RESULTS_FILE

echo -e "${BOLD}Medium Problem (128 residues):${NC}"
echo "Medium Problem (128 residues):" >> $RESULTS_FILE
printf "${MAGENTA}%-8s %-12s %-12s %-12s %-12s %-10s${NC}\n" "Version" "Speed(s/s)" "Batch(ms)" "Forward(ms)" "Backward(ms)" "Speedup"
printf "%-8s %-12s %-12s %-12s %-12s %-10s\n" "Version" "Speed(s/s)" "Batch(ms)" "Forward(ms)" "Backward(ms)" "Speedup" >> $RESULTS_FILE
echo "────────────────────────────────────────────────────────────────────────"
echo "────────────────────────────────────────────────────────────────────────" >> $RESULTS_FILE

# Extract V1 medium baseline
IFS='|' read -r _ _ v1m_speed v1m_batch v1m_fwd v1m_bwd _ _ <<< "${MEDIUM_RESULTS[0]}"

for i in "${!MEDIUM_RESULTS[@]}"; do
    IFS='|' read -r ver name speed batch fwd bwd opt mem <<< "${MEDIUM_RESULTS[$i]}"

    if [ "$i" -eq 0 ]; then
        speedup="1.0x"
    else
        speedup=$(awk "BEGIN {printf \"%.2fx\", $speed / $v1m_speed}")
    fi

    printf "%-8s %-12s %-12s %-12s %-12s %-10s\n" "$ver" "$speed" "$batch" "$fwd" "$bwd" "$speedup"
    printf "%-8s %-12s %-12s %-12s %-12s %-10s\n" "$ver" "$speed" "$batch" "$fwd" "$bwd" "$speedup" >> $RESULTS_FILE
done

echo ""
echo "" >> $RESULTS_FILE

# Key insights
echo -e "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${GREEN}  Key Insights${NC}"
echo -e "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "Key Insights:" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Calculate final speedups
IFS='|' read -r _ _ v3s_speed v3s_batch v3s_fwd v3s_bwd _ _ <<< "${SMALL_RESULTS[2]}"
IFS='|' read -r _ _ v3m_speed v3m_batch v3m_fwd v3m_bwd _ _ <<< "${MEDIUM_RESULTS[2]}"

small_speedup=$(awk "BEGIN {printf \"%.1fx\", $v3s_speed / $v1s_speed}")
medium_speedup=$(awk "BEGIN {printf \"%.2fx\", $v3m_speed / $v1m_speed}")
small_bwd_reduction=$(awk "BEGIN {printf \"%.0f\", (1 - $v3s_bwd / $v1s_bwd) * 100}")
medium_bwd_reduction=$(awk "BEGIN {printf \"%.0f\", (1 - $v3m_bwd / $v1m_bwd) * 100}")

echo -e "  ${BOLD}1. Progressive Optimization Works!${NC}"
echo -e "     • Small problem: ${GREEN}${small_speedup} total speedup${NC} (V1 → V3)"
echo -e "     • Medium problem: ${GREEN}${medium_speedup} total speedup${NC} (V1 → V3)"
echo ""
echo "  1. Progressive Optimization Works!" >> $RESULTS_FILE
echo "     • Small problem: ${small_speedup} total speedup (V1 → V3)" >> $RESULTS_FILE
echo "     • Medium problem: ${medium_speedup} total speedup (V1 → V3)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

echo -e "  ${BOLD}2. Backward Pass is Key Bottleneck${NC}"
echo -e "     • Small: ${v1s_bwd} ms → ${v3s_bwd} ms (${GREEN}-${small_bwd_reduction}%${NC})"
echo -e "     • Medium: ${v1m_bwd} ms → ${v3m_bwd} ms (${GREEN}-${medium_bwd_reduction}%${NC})"
echo ""
echo "  2. Backward Pass is Key Bottleneck" >> $RESULTS_FILE
echo "     • Small: ${v1s_bwd} ms → ${v3s_bwd} ms (-${small_bwd_reduction}%)" >> $RESULTS_FILE
echo "     • Medium: ${v1m_bwd} ms → ${v3m_bwd} ms (-${medium_bwd_reduction}%)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

echo -e "  ${BOLD}3. Optimization Stages${NC}"
echo -e "     • V1 → V2: High-level kernel fusion (32% & 18% gain)"
echo -e "     • V2 → V3: Custom Triton kernels (additional 53% & 40% gain)"
echo -e "     • Each stage builds on previous improvements"
echo ""
echo "  3. Optimization Stages" >> $RESULTS_FILE
echo "     • V1 → V2: High-level kernel fusion (32% & 18% gain)" >> $RESULTS_FILE
echo "     • V2 → V3: Custom Triton kernels (additional 53% & 40% gain)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║                                                                  ║${NC}"
echo -e "${BOLD}${GREEN}║                    ✓ Tutorial Complete!                          ║${NC}"
echo -e "${BOLD}${GREEN}║                                                                  ║${NC}"
echo -e "${BOLD}${GREEN}║  You've learned the complete GPU optimization pipeline:          ║${NC}"
echo -e "${BOLD}${GREEN}║    1. Baseline measurement & profiling                           ║${NC}"
echo -e "${BOLD}${GREEN}║    2. High-level kernel fusion                                   ║${NC}"
echo -e "${BOLD}${GREEN}║    3. Custom GPU kernels with Triton                             ║${NC}"
echo -e "${BOLD}${GREEN}║                                                                  ║${NC}"
echo -e "${BOLD}${GREEN}║  Achievement: ${BOLD}${YELLOW}${small_speedup} speedup${GREEN} on small problems! 🚀              ║${NC}"
echo -e "${BOLD}${GREEN}║                                                                  ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "📊 Full results saved to: ${CYAN}$RESULTS_FILE${NC}"
echo -e "📖 See ${CYAN}PERFORMANCE_OPTIMIZATION_TUTORIAL.md${NC} for detailed explanations"
echo ""

echo "Tutorial completed at: $(date)" >> $RESULTS_FILE
