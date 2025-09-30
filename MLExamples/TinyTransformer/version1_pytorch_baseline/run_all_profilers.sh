#!/bin/bash

# Comprehensive Profiling Suite for Tiny LLaMA V1 Baseline
# This script orchestrates all profiling tools for complete performance analysis

set -e  # Exit on error

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

log_analysis() {
    echo -e "${PURPLE}[ANALYSIS]${NC} $1"
}

# Default configuration
BATCH_SIZE=8
SEQ_LEN=128
NUM_STEPS=50
PROFILE_DIR="./complete_analysis_$(date +%Y%m%d_%H%M%S)"
PYTORCH_STEPS=10
FLOPS_STEPS=20
GENERATE_REPORTS=true
VALIDATE_ENV=true

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
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --profile-dir)
            PROFILE_DIR="$2"
            shift 2
            ;;
        --pytorch-steps)
            PYTORCH_STEPS="$2"
            shift 2
            ;;
        --flops-steps)
            FLOPS_STEPS="$2"
            shift 2
            ;;
        --no-reports)
            GENERATE_REPORTS=false
            shift
            ;;
        --skip-validation)
            VALIDATE_ENV=false
            shift
            ;;
        --help|-h)
            echo "Comprehensive Profiling Suite for Tiny LLaMA V1"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --batch-size SIZE      Batch size (default: 8)"
            echo "  --seq-len LENGTH       Sequence length (default: 128)"
            echo "  --num-steps STEPS      Training steps (default: 50)"
            echo "  --profile-dir DIR      Profile output directory"
            echo "  --pytorch-steps STEPS  PyTorch profiling steps (default: 10)"
            echo "  --flops-steps STEPS    FLOPS profiling steps (default: 20)"
            echo "  --no-reports           Skip report generation"
            echo "  --skip-validation      Skip environment validation"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run with defaults"
            echo "  $0 --batch-size 16 --seq-len 256     # Custom batch and sequence"
            echo "  $0 --profile-dir ./my_analysis        # Custom output directory"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print banner
echo "=" * 80
echo "CASTILLE AI WORKSHOP - COMPREHENSIVE PROFILING SUITE"
echo "     Version 1: PyTorch Baseline Performance Analysis"
echo "=" * 80
echo ""

log_info "Configuration:"
log_info "  Batch size: $BATCH_SIZE"
log_info "  Sequence length: $SEQ_LEN"
log_info "  Training steps: $NUM_STEPS"
log_info "  Profile directory: $PROFILE_DIR"
log_info "  PyTorch profiling steps: $PYTORCH_STEPS"
log_info "  FLOPS profiling steps: $FLOPS_STEPS"
log_info "  Generate reports: $GENERATE_REPORTS"

# Create profile directory
mkdir -p "$PROFILE_DIR"
cd "$PROFILE_DIR"
log_info "Created profile directory: $(pwd)"

# Environment validation
if [ "$VALIDATE_ENV" = true ]; then
    log_step "1. Environment Validation"

    # Check Python environment
    if ! command -v python &> /dev/null; then
        log_error "Python not found. Please ensure Python is installed and in PATH."
        exit 1
    fi

    # Check if we're in the right directory
    if [ ! -f "../tiny_llama_v1.py" ]; then
        log_error "tiny_llama_v1.py not found. Please run this script from version1_pytorch_baseline directory."
        exit 1
    fi

    # Quick environment test
    log_info "Running environment validation..."
    python ../tiny_llama_v1.py --validate-setup --batch-size 2 --num-steps 2 > validation.log 2>&1

    if [ $? -eq 0 ]; then
        log_info "Environment validation passed"
        rm -f validation.log
    else
        log_error "Environment validation failed. Check validation.log for details."
        cat validation.log
        exit 1
    fi
else
    log_warning "Skipping environment validation"
fi

# Step 2: Baseline Training Run
log_step "2. Baseline Training Run"
log_info "Running baseline training to establish performance metrics..."

python ../tiny_llama_v1.py \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --num-steps $NUM_STEPS \
    > baseline_training.log 2>&1

if [ $? -eq 0 ]; then
    log_info "Baseline training completed"
    grep -E "(samples/sec|Loss:|Memory:)" baseline_training.log | tail -5
else
    log_error "Baseline training failed"
    tail -20 baseline_training.log
    exit 1
fi

# Step 3: PyTorch Profiler Analysis
log_step "3. PyTorch Profiler Analysis"
log_info "Running comprehensive PyTorch profiling..."

PYTORCH_DIR="pytorch_profiling"
mkdir -p "$PYTORCH_DIR"

python ../run_pytorch_profiler.py \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --num-steps $PYTORCH_STEPS \
    --profile-dir "$PYTORCH_DIR" \
    --include-memory \
    --include-shapes \
    --generate-report \
    > pytorch_profiling.log 2>&1

if [ $? -eq 0 ]; then
    log_info "PyTorch profiling completed"
    log_info "📁 Results saved to: $PYTORCH_DIR"
else
    log_warning "PyTorch profiling had issues (check pytorch_profiling.log)"
    tail -10 pytorch_profiling.log
fi

# Step 4: DeepSpeed FLOPS Analysis
log_step "4. DeepSpeed FLOPS Analysis"
log_info "Running FLOPS profiling and computational intensity analysis..."

FLOPS_DIR="flops_analysis"
mkdir -p "$FLOPS_DIR"

python ../run_deepspeed_flops.py \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --num-steps $FLOPS_STEPS \
    --output-dir "$FLOPS_DIR" \
    --detailed-analysis \
    --computational-intensity \
    --generate-roofline \
    > flops_analysis.log 2>&1

if [ $? -eq 0 ]; then
    log_info "FLOPS analysis completed"
    log_info "📁 Results saved to: $FLOPS_DIR"

    # Extract key metrics
    if [ -f "$FLOPS_DIR/flops_profile.json" ]; then
        MFU=$(python -c "import json; data=json.load(open('$FLOPS_DIR/flops_profile.json')); print(f\"{data['efficiency_metrics']['mfu_percent']:.1f}%\")" 2>/dev/null || echo "N/A")
        THROUGHPUT=$(python -c "import json; data=json.load(open('$FLOPS_DIR/flops_profile.json')); print(f\"{data['performance_metrics']['throughput_samples_per_sec']:.1f}\")" 2>/dev/null || echo "N/A")
        log_info "Model FLOPS Utilization: $MFU"
        log_info "Throughput: $THROUGHPUT samples/sec"
    fi
else
    log_warning "FLOPS analysis had issues (check flops_analysis.log)"
    tail -10 flops_analysis.log
fi

# Step 5: Memory Analysis
log_step "5. Memory Usage Analysis"
log_info "Analyzing memory usage patterns..."

# Run memory-focused profiling
python ../tiny_llama_v1.py \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --num-steps 20 \
    --enable-pytorch-profiler \
    --enable-memory-profiling \
    --profile-dir "./memory_analysis" \
    > memory_analysis.log 2>&1

if [ $? -eq 0 ]; then
    log_info "Memory analysis completed"

    # Extract memory usage
    if command -v nvidia-smi &> /dev/null; then
        log_info "Current GPU memory usage:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1
    elif command -v rocm-smi &> /dev/null; then
        log_info "Current GPU memory usage:"
        rocm-smi --showmeminfo vram | grep "VRAM Total\\|VRAM Used"
    fi
else
    log_warning "Memory analysis had issues"
fi

# Step 6: Generate Comprehensive Reports
if [ "$GENERATE_REPORTS" = true ]; then
    log_step "6. Generating Comprehensive Reports"
    log_info "Creating analysis reports and summaries..."

    # Create summary report
    SUMMARY_REPORT="performance_summary_report.md"

    cat > "$SUMMARY_REPORT" << EOF
# Tiny LLaMA V1 Baseline - Performance Analysis Report

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')
**Configuration:** Batch size $BATCH_SIZE, Sequence length $SEQ_LEN, Steps $NUM_STEPS

## Executive Summary

This report provides comprehensive performance analysis of the Tiny LLaMA V1 baseline implementation,
establishing the foundation for optimization work in subsequent workshop versions.

## Configuration Details

- **Model Configuration:**
  - Hidden dimension: 256
  - Number of layers: 4
  - Number of attention heads: 8
  - Sequence length: $SEQ_LEN
  - Vocabulary size: 1000

- **Training Configuration:**
  - Batch size: $BATCH_SIZE
  - Training steps: $NUM_STEPS
  - Device: $(python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>/dev/null || echo "Unknown")

## Performance Metrics

### Baseline Training Performance
EOF

    # Add baseline metrics if available
    if [ -f "baseline_training.log" ]; then
        echo -e "\n\`\`\`" >> "$SUMMARY_REPORT"
        grep -E "(samples/sec|Loss:|Memory:|Time:)" baseline_training.log | tail -10 >> "$SUMMARY_REPORT"
        echo -e "\`\`\`\n" >> "$SUMMARY_REPORT"
    fi

    # Add FLOPS metrics if available
    if [ -f "$FLOPS_DIR/flops_profile.json" ]; then
        cat >> "$SUMMARY_REPORT" << EOF
### FLOPS Analysis Results

- **Model FLOPS Utilization:** $MFU
- **Training Throughput:** $THROUGHPUT samples/sec
- **Computational Analysis:** See \`$FLOPS_DIR/flops_profile.json\`

EOF
    fi

    # Add recommendations
    cat >> "$SUMMARY_REPORT" << EOF
## Optimization Recommendations for Version 2

Based on this baseline analysis, the following optimizations are recommended:

### High Priority
1. **QKV Fusion** - Combine separate Q, K, V linear projections to reduce kernel launch overhead
2. **Flash Attention** - Implement memory-efficient attention computation
3. **SwiGLU Fusion** - Merge gate and up projections in feed-forward network

### Medium Priority
4. **Kernel Fusion** - Reduce GPU kernel launch overhead through operation fusion
5. **Memory Layout Optimization** - Improve memory access patterns
6. **Mixed Precision** - Consider FP16/BF16 for additional speedup

## Analysis Files

- **Baseline Training:** \`baseline_training.log\`
- **PyTorch Profiling:** \`$PYTORCH_DIR/\`
- **FLOPS Analysis:** \`$FLOPS_DIR/\`
- **Memory Analysis:** \`memory_analysis/\`

## Next Steps

1. Review detailed profiling results in respective directories
2. Use TensorBoard for visualization: \`tensorboard --logdir $PYTORCH_DIR\`
3. Proceed to Version 2 for kernel fusion optimizations

---
*Generated by Castille AI Workshop Profiling Suite*
EOF

    log_info "Summary report generated: $SUMMARY_REPORT"

    # Create analysis summary
    log_analysis "Analysis Summary:"
    echo "  Profiling Results:"
    echo "     - Baseline training: $([ -f baseline_training.log ] && echo "PASS" || echo "FAIL")"
    echo "     - PyTorch profiling: $([ -d "$PYTORCH_DIR" ] && echo "PASS" || echo "FAIL")"
    echo "     - FLOPS analysis: $([ -d "$FLOPS_DIR" ] && echo "PASS" || echo "FAIL")"
    echo "     - Memory analysis: $([ -d "memory_analysis" ] && echo "PASS" || echo "FAIL")"
fi

# Step 7: Final Summary and Next Steps
log_step "7. Analysis Complete"

echo ""
echo "Comprehensive profiling analysis completed!"
echo ""
echo "📁 Results Location: $(pwd)"
echo ""
echo "Key Files Generated:"
echo "   - performance_summary_report.md    # Executive summary"
echo "   - baseline_training.log            # Baseline performance"
echo "   - $PYTORCH_DIR/                    # PyTorch profiler results"
echo "   - $FLOPS_DIR/                      # FLOPS and efficiency analysis"
echo "   - memory_analysis/                 # Memory usage patterns"
echo ""
echo "Next Steps:"
echo "   1. Review the summary report: cat performance_summary_report.md"
echo "   2. Launch TensorBoard: tensorboard --logdir $PYTORCH_DIR"
echo "   3. Analyze bottlenecks and optimization opportunities"
echo "   4. Proceed to Version 2 for kernel fusion optimizations"
echo ""
echo "Quick Analysis Commands:"
echo "   # View PyTorch profiling summary"
echo "   cat $PYTORCH_DIR/comprehensive_profiling_report.md"
echo ""
echo "   # View FLOPS analysis"
echo "   python -c \"import json; print(json.dumps(json.load(open('$FLOPS_DIR/flops_profile.json')), indent=2))\""
echo ""
echo "   # Compare with Version 2 later:"
echo "   cd ../version2_pytorch_fused && bash run_all_profilers.sh --profile-dir ../comparison_v2"
echo ""

log_info "Profile data saved to: $(pwd)"
log_info "Workshop Version 1 analysis complete! Ready for Version 2 optimizations."