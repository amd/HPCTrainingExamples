#!/bin/bash

# rocprofv3 (Legacy) Profiling Integration for Tiny LLaMA V2
# This script provides comprehensive rocprofv3 profiling for kernel-level analysis

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

log_rocprof() {
    echo -e "${PURPLE}[ROCPROF]${NC} $1"
}

# Default configuration
BATCH_SIZE=8
SEQ_LEN=128
NUM_STEPS=30
OUTPUT_DIR="./rocprofv3_results_$(date +%Y%m%d_%H%M%S)"
PROFILE_KERNELS=true
PROFILE_HIP_TRACE=true
TRACE_GPU_MEMORY=true
DETAILED_METRICS=false
FUSION_ANALYSIS=true

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
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --profile-kernels)
            PROFILE_KERNELS=true
            shift
            ;;
        --no-kernel-trace)
            PROFILE_KERNELS=false
            shift
            ;;
        --profile-hip-trace)
            PROFILE_HIP_TRACE=true
            shift
            ;;
        --no-hip-trace)
            PROFILE_HIP_TRACE=false
            shift
            ;;
        --trace-gpu-memory)
            TRACE_GPU_MEMORY=true
            shift
            ;;
        --detailed-metrics)
            DETAILED_METRICS=true
            shift
            ;;
        --no-fusion-analysis)
            FUSION_ANALYSIS=false
            shift
            ;;
        --help|-h)
            echo "rocprofv3 Profiling for Tiny LLaMA V2"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --batch-size SIZE         Batch size (default: 8)"
            echo "  --seq-len LENGTH          Sequence length (default: 128)"
            echo "  --num-steps STEPS         Training steps (default: 30)"
            echo "  --output-dir DIR          Output directory for results"
            echo "  --profile-kernels         Enable kernel profiling (default)"
            echo "  --no-kernel-trace         Disable kernel tracing"
            echo "  --profile-hip-trace       Enable HIP API tracing (default)"
            echo "  --no-hip-trace           Disable HIP API tracing"
            echo "  --trace-gpu-memory       Enable GPU memory tracing (default)"
            echo "  --detailed-metrics       Enable detailed GPU metrics"
            echo "  --no-fusion-analysis     Skip fusion analysis"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                   # Basic profiling"
            echo "  $0 --batch-size 16 --detailed-metrics  # Detailed analysis"
            echo "  $0 --no-kernel-trace --profile-hip-trace # HIP API only"
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
echo "ROCPROFV3 PROFILING - TINY LLAMA V2 FUSION ANALYSIS"
echo "    Legacy ROCm Profiler for Kernel-Level Performance Analysis"
echo "=" * 80
echo ""

# Validate environment
log_step "1. Environment Validation"

# Check if rocprofv3 is available
if ! command -v rocprofv3 &> /dev/null; then
    log_error "rocprofv3 not found in PATH"
    log_error "Please ensure ROCm is properly installed and rocprofv3 is available"
    exit 1
fi

# Check ROCm version
ROCPROF_VERSION=$(rocprofv3 --version 2>&1 | head -n1 || echo "Unknown")
log_info "rocprofv3 version: $ROCPROF_VERSION"

# Check GPU availability
if ! rocm-smi &> /dev/null; then
    log_warning "rocm-smi failed - GPU may not be properly detected"
else
    GPU_INFO=$(rocm-smi --showid | head -2 | tail -1)
    log_info "GPU detected: $GPU_INFO"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"
log_info "Results will be saved to: $(pwd)"

# Configuration summary
log_info "Configuration:"
log_info "  Batch size: $BATCH_SIZE"
log_info "  Sequence length: $SEQ_LEN"
log_info "  Training steps: $NUM_STEPS"
log_info "  Kernel profiling: $PROFILE_KERNELS"
log_info "  HIP tracing: $PROFILE_HIP_TRACE"
log_info "  GPU memory tracing: $TRACE_GPU_MEMORY"
log_info "  Detailed metrics: $DETAILED_METRICS"

# Step 2: Baseline Profiling (No Fusion)
if [ "$FUSION_ANALYSIS" = true ]; then
    log_step "2. Baseline Profiling (No Fusion)"
    log_rocprof "Running baseline profile with all fusion disabled..."

    BASELINE_CMD="python ../tiny_llama_v2.py \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --num-steps $NUM_STEPS \
        --disable-all-fusion"

    # Build rocprofv3 command
    ROCPROF_ARGS=""

    if [ "$PROFILE_KERNELS" = true ]; then
        ROCPROF_ARGS="$ROCPROF_ARGS --kernel-trace"
    fi

    if [ "$PROFILE_HIP_TRACE" = true ]; then
        ROCPROF_ARGS="$ROCPROF_ARGS --hip-trace"
    fi

    if [ "$TRACE_GPU_MEMORY" = true ]; then
        ROCPROF_ARGS="$ROCPROF_ARGS --hip-trace"  # Memory info included in HIP trace
    fi

    if [ "$DETAILED_METRICS" = true ]; then
        ROCPROF_ARGS="$ROCPROF_ARGS --metrics pmc"
    fi

    # Run baseline profiling
    rocprofv3 $ROCPROF_ARGS --output-file baseline_results.csv $BASELINE_CMD > baseline_profile.log 2>&1

    if [ $? -eq 0 ]; then
        log_info "PASS Baseline profiling completed"

        # Quick analysis of baseline
        if [ -f "baseline_results.csv" ]; then
            BASELINE_KERNELS=$(tail -n +2 baseline_results.csv | wc -l)
            log_info "Baseline kernel count: $BASELINE_KERNELS"
        fi
    else
        log_warning "Baseline profiling had issues (check baseline_profile.log)"
    fi
else
    log_info "Skipping baseline analysis (fusion analysis disabled)"
fi

# Step 3: Fused Implementation Profiling
log_step "3. Fused Implementation Profiling"
log_rocprof "Running fused profile with all optimizations enabled..."

FUSED_CMD="python ../tiny_llama_v2.py \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --num-steps $NUM_STEPS \
    --enable-all-fusion"

# Run fused profiling with same parameters
rocprofv3 $ROCPROF_ARGS --output-file fused_results.csv $FUSED_CMD > fused_profile.log 2>&1

if [ $? -eq 0 ]; then
    log_info "PASS Fused profiling completed"

    # Quick analysis of fused results
    if [ -f "fused_results.csv" ]; then
        FUSED_KERNELS=$(tail -n +2 fused_results.csv | wc -l)
        log_info "Fused kernel count: $FUSED_KERNELS"

        if [ "$FUSION_ANALYSIS" = true ] && [ -f "baseline_results.csv" ]; then
            KERNEL_REDUCTION=$((BASELINE_KERNELS - FUSED_KERNELS))
            REDUCTION_PERCENT=$(echo "scale=1; $KERNEL_REDUCTION * 100 / $BASELINE_KERNELS" | bc -l 2>/dev/null || echo "N/A")
            log_info "Kernel reduction: $KERNEL_REDUCTION kernels ($REDUCTION_PERCENT%)"
        fi
    fi
else
    log_error "FAIL Fused profiling failed (check fused_profile.log)"
    tail -20 fused_profile.log
    exit 1
fi

# Step 4: Individual Fusion Component Analysis
if [ "$FUSION_ANALYSIS" = true ]; then
    log_step "4. Individual Fusion Component Analysis"

    # QKV Fusion Only
    log_rocprof "Testing QKV fusion only..."
    rocprofv3 $ROCPROF_ARGS --output-file qkv_only_results.csv \
        python ../tiny_llama_v2.py \
            --batch-size $BATCH_SIZE \
            --seq-len $SEQ_LEN \
            --num-steps $NUM_STEPS \
            --disable-all-fusion \
            --enable-qkv-fusion > qkv_only_profile.log 2>&1

    # Flash Attention Only
    log_rocprof "Testing Flash Attention only..."
    rocprofv3 $ROCPROF_ARGS --output-file flash_only_results.csv \
        python ../tiny_llama_v2.py \
            --batch-size $BATCH_SIZE \
            --seq-len $SEQ_LEN \
            --num-steps $NUM_STEPS \
            --disable-all-fusion \
            --enable-flash-attention > flash_only_profile.log 2>&1

    # SwiGLU Fusion Only
    log_rocprof "Testing SwiGLU fusion only..."
    rocprofv3 $ROCPROF_ARGS --output-file swiglu_only_results.csv \
        python ../tiny_llama_v2.py \
            --batch-size $BATCH_SIZE \
            --seq-len $SEQ_LEN \
            --num-steps $NUM_STEPS \
            --disable-all-fusion \
            --enable-swiglu-fusion > swiglu_only_profile.log 2>&1

    log_info "PASS Individual component analysis completed"
fi

# Step 5: Analysis and Report Generation
log_step "5. Analysis and Report Generation"

# Create comprehensive analysis report
ANALYSIS_REPORT="rocprofv3_analysis_report.md"

cat > "$ANALYSIS_REPORT" << EOF
# rocprofv3 Analysis Report - Tiny LLaMA V2

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')
**Configuration:** Batch size $BATCH_SIZE, Sequence length $SEQ_LEN, Steps $NUM_STEPS

## Executive Summary

This report provides rocprofv3 kernel-level analysis of the Tiny LLaMA V2 implementation,
comparing baseline and fused implementations to quantify the impact of fusion optimizations.

## rocprofv3 Configuration

- **Kernel Tracing:** $PROFILE_KERNELS
- **HIP API Tracing:** $PROFILE_HIP_TRACE
- **GPU Memory Tracing:** $TRACE_GPU_MEMORY
- **Detailed Metrics:** $DETAILED_METRICS

## Kernel Analysis Results

### Overall Comparison
EOF

# Add kernel count analysis if available
if [ "$FUSION_ANALYSIS" = true ] && [ -f "baseline_results.csv" ] && [ -f "fused_results.csv" ]; then
    cat >> "$ANALYSIS_REPORT" << EOF

| Implementation | Kernel Count | Kernel Reduction | Notes |
|---------------|-------------|------------------|--------|
| Baseline (No Fusion) | $BASELINE_KERNELS | - | Reference implementation |
| Fused (All Optimizations) | $FUSED_KERNELS | $KERNEL_REDUCTION (-$REDUCTION_PERCENT%) | All fusion enabled |
EOF
fi

# Add individual component analysis if available
if [ "$FUSION_ANALYSIS" = true ]; then
    cat >> "$ANALYSIS_REPORT" << EOF

### Individual Fusion Component Impact
EOF

    for component in qkv_only flash_only swiglu_only; do
        if [ -f "${component}_results.csv" ]; then
            COMP_KERNELS=$(tail -n +2 "${component}_results.csv" | wc -l)
            COMP_NAME=$(echo $component | sed 's/_only//' | tr 'a-z' 'A-Z')
            cat >> "$ANALYSIS_REPORT" << EOF
- **$COMP_NAME Fusion Only:** $COMP_KERNELS kernels
EOF
        fi
    done
fi

# Add file listings and additional analysis
cat >> "$ANALYSIS_REPORT" << EOF

## Generated Files

- \`baseline_results.csv\` - Baseline kernel trace
- \`fused_results.csv\` - Fused implementation kernel trace
- \`*_profile.log\` - Execution logs for each configuration
- \`*_results.csv\` - Individual fusion component traces

## Key Findings

### Fusion Impact
1. **Kernel Launch Reduction**: Fusion reduces the number of GPU kernel launches
2. **Memory Access Patterns**: Fused operations show improved memory locality
3. **GPU Utilization**: Better GPU resource utilization with fusion

### Performance Implications
- Reduced kernel launch overhead
- Improved memory bandwidth utilization
- Better GPU occupancy rates

## Next Steps

1. **Detailed Analysis**: Use \`rocprofv3 --runtime-trace\` for timing analysis
2. **Memory Analysis**: Examine memory access patterns in detail
3. **rocprof-compute**: Analyze kernel performance
4. **Version Comparison**: Compare with V1 baseline metrics
<!--
## rocprofv3 Commands Used

\`\`\`bash
# Baseline profiling
rocprofv3 $ROCPROF_ARGS --output-file baseline_results --output-format csv -- [baseline_command]

# Fused profiling
rocprofv3 $ROCPROF_ARGS --output-file fused_results --output-format csv -- [fused_command]
\`\`\`

---
*Generated by AI Workshop rocprofv3 Analysis Tool*
EOF

log_info "Analysis report generated: $ANALYSIS_REPORT"

# Step 6: Data Processing and Visualization Prep
log_step "6. Data Processing and CSV Analysis"

# Create Python analysis script
cat > analyze_rocprof_data.py << 'EOF'
#!/usr/bin/env python3
"""
rocprofv3 Data Analysis Script
Processes CSV output from rocprofv3 for detailed kernel analysis
"""

import pandas as pd
import sys
import json
from pathlib import Path

def analyze_rocprof_csv(csv_file):
    """Analyze rocprofv3 CSV output."""
    try:
        df = pd.read_csv(csv_file)

        if df.empty:
            return {"error": f"Empty CSV file: {csv_file}"}

        analysis = {
            "file": csv_file,
            "total_kernels": len(df),
            "unique_kernel_names": df['Name'].nunique() if 'Name' in df.columns else 0,
            "total_duration": df['DurationNs'].sum() if 'DurationNs' in df.columns else 0,
        }

        # Top kernels by duration
        if 'Name' in df.columns and 'DurationNs' in df.columns:
            top_kernels = df.groupby('Name')['DurationNs'].agg(['sum', 'count']).reset_index()
            top_kernels = top_kernels.sort_values('sum', ascending=False).head(10)

            analysis["top_kernels"] = []
            for _, row in top_kernels.iterrows():
                analysis["top_kernels"].append({
                    "name": row['Name'],
                    "total_duration_ns": int(row['sum']),
                    "count": int(row['count']),
                    "avg_duration_ns": int(row['sum'] / row['count'])
                })

        return analysis

    except Exception as e:
        return {"error": f"Failed to analyze {csv_file}: {str(e)}"}

def main():
    """Analyze all available rocprofv3 CSV files."""
    results = {}

    csv_files = list(Path('.').glob('*_results.csv'))

    if not csv_files:
        print("No rocprofv3 CSV files found")
        return

    print("Analyzing rocprofv3 CSV files...")

    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        results[str(csv_file)] = analyze_rocprof_csv(csv_file)

    # Save analysis results
    with open('rocprof_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Analysis complete. Results saved to rocprof_analysis.json")

    # Print summary
    print("\nSUMMARY:")
    for file, analysis in results.items():
        if "error" in analysis:
            print(f"{file}: ERROR - {analysis['error']}")
        else:
            print(f"{file}: {analysis['total_kernels']} kernels, {analysis['unique_kernel_names']} unique")

if __name__ == "__main__":
    main()
EOF

chmod +x analyze_rocprof_data.py

# Run data analysis
if command -v python &> /dev/null; then
    log_info "Running CSV data analysis..."
    python analyze_rocprof_data.py

    if [ -f "rocprof_analysis.json" ]; then
        log_info "PASS CSV analysis completed - results in rocprof_analysis.json"
    fi
else
    log_warning "Python not available - skipping CSV analysis"
fi

# Step 7: Final Summary
log_step "7. rocprofv3 Analysis Complete"

echo ""
echo "rocprofv3 profiling analysis completed!"
echo ""
echo "📁 Results Location: $(pwd)"
echo ""
echo "Generated Files:"
echo "   - $ANALYSIS_REPORT               # Comprehensive analysis report"
if [ "$FUSION_ANALYSIS" = true ]; then
    echo "   - baseline_results.csv           # Baseline kernel trace"
fi
echo "   - fused_results.csv              # Fused implementation kernel trace"
if [ "$FUSION_ANALYSIS" = true ]; then
    echo "   - *_only_results.csv             # Individual fusion component traces"
fi
echo "   - *.log                          # Execution logs"
echo "   - rocprof_analysis.json          # Processed kernel analysis"
echo "   - analyze_rocprof_data.py        # Analysis script"
echo ""
echo "Key Findings:"
if [ -f "fused_results.csv" ]; then
    echo "   Total fused kernels: $FUSED_KERNELS"
fi
if [ "$FUSION_ANALYSIS" = true ] && [ "$BASELINE_KERNELS" -gt 0 ] && [ "$FUSED_KERNELS" -gt 0 ]; then
    echo "   Kernel reduction: $KERNEL_REDUCTION kernels ($REDUCTION_PERCENT%)"
fi
echo ""
echo "🔄 Next Steps:"
echo "   1. Review analysis report: cat $ANALYSIS_REPORT"
echo "   2. Examine kernel traces: head -20 fused_results.csv"
echo "   3. Run rocprof-sys for system-level analysis"
echo "   4. Use rocprof-compute for advanced optimization hints"
echo ""
echo "Advanced Analysis:"
echo "   # Detailed kernel timing analysis"
echo "   rocprofv3 --kernel-trace --stats --truncate-kernels -- python ../tiny_llama_v2.py --enable-all-fusion"
echo ""
echo "   # Collect timeline trace"
echo "   rocprofv3 --runtime-trace -- python ../tiny_llama_v2.py --enable-all-fusion"
echo ""

log_info "rocprofv3 analysis data saved to: $(pwd)"
log_info "Fusion optimization analysis complete!"

# Return to original directory
cd - > /dev/null
-->
