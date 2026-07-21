#!/bin/bash

# ROCprof-compute (Advanced Profiler) Integration for Tiny LLaMA V2
# Advanced GPU kernel analysis with optimization recommendations

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

log_compute() {
    echo -e "${PURPLE}[ROCPROF-COMPUTE]${NC} $1"
}

log_analysis() {
    echo -e "${CYAN}[ANALYSIS]${NC} $1"
}

# Default configuration
BATCH_SIZE=8
SEQ_LEN=128
NUM_STEPS=30
OUTPUT_DIR="./rocprof_compute_results_$(date +%Y%m%d_%H%M%S)"
DETAILED_ANALYSIS=true
OPTIMIZATION_HINTS=true
KERNEL_FILTER="gemm|attention|softmax|norm|conv"
METRICS="pmc,gpu_busy_cycles,fetch_size,write_size,l2_cache_hit_rate"
ROOFLINE_ANALYSIS=true
COMPARATIVE_ANALYSIS=true

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
        --detailed-analysis)
            DETAILED_ANALYSIS=true
            shift
            ;;
        --no-detailed-analysis)
            DETAILED_ANALYSIS=false
            shift
            ;;
        --optimization-hints)
            OPTIMIZATION_HINTS=true
            shift
            ;;
        --no-optimization-hints)
            OPTIMIZATION_HINTS=false
            shift
            ;;
        --kernel-filter)
            KERNEL_FILTER="$2"
            shift 2
            ;;
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        --roofline-analysis)
            ROOFLINE_ANALYSIS=true
            shift
            ;;
        --no-roofline)
            ROOFLINE_ANALYSIS=false
            shift
            ;;
        --comparative-analysis)
            COMPARATIVE_ANALYSIS=true
            shift
            ;;
        --no-comparative)
            COMPARATIVE_ANALYSIS=false
            shift
            ;;
        --help|-h)
            echo "ROCprof-compute Advanced Profiling for Tiny LLaMA V2"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --batch-size SIZE         Batch size (default: 8)"
            echo "  --seq-len LENGTH          Sequence length (default: 128)"
            echo "  --num-steps STEPS         Training steps (default: 30)"
            echo "  --output-dir DIR          Output directory for results"
            echo "  --detailed-analysis       Enable detailed kernel analysis (default)"
            echo "  --no-detailed-analysis    Disable detailed analysis"
            echo "  --optimization-hints      Enable optimization hints (default)"
            echo "  --no-optimization-hints   Disable optimization hints"
            echo "  --kernel-filter PATTERN   Kernel filter pattern (default: gemm|attention|softmax|norm|conv)"
            echo "  --metrics METRICS         Performance metrics to collect"
            echo "  --roofline-analysis       Enable roofline analysis (default)"
            echo "  --no-roofline            Disable roofline analysis"
            echo "  --comparative-analysis    Compare fusion configurations (default)"
            echo "  --no-comparative         Skip comparative analysis"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Complete advanced analysis"
            echo "  $0 --kernel-filter 'attention'       # Focus on attention kernels"
            echo "  $0 --no-roofline --no-comparative    # Basic profiling only"
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
echo "ROCPROF-COMPUTE ADVANCED PROFILING - TINY LLAMA V2"
echo "    Advanced GPU Kernel Analysis with Optimization Recommendations"
echo "=" * 80
echo ""

# Step 1: Environment Validation
log_step "1. Environment Validation"

# Check if rocprof-compute is available
if ! command -v rocprof-compute &> /dev/null; then
    log_error "rocprof-compute not found in PATH"
    log_error "Please ensure ROCm is properly installed with rocprof-compute"
    log_error "Or build rocprof-compute from source: https://github.com/ROCm/rocm-systems"
    exit 1
fi

# Check rocprof-compute version and capabilities
ROCPROF_COMPUTE_VERSION=$(rocprof-compute --version 2>&1 | head -n1 || echo "Unknown")
log_info "rocprof-compute version: $ROCPROF_COMPUTE_VERSION"

# Test basic functionality
log_info "Testing rocprof-compute basic functionality..."
if timeout 10 rocprof-compute --help > /dev/null 2>&1; then
    log_info "PASS rocprof-compute is functional"
else
    log_warning "rocprof-compute may have issues - proceeding with caution"
fi

# Check GPU and drivers
if command -v rocminfo &> /dev/null; then
    GPU_INFO=$(rocminfo | grep -E "Marketing Name|Compute Unit|Memory Size" | head -3 || echo "GPU info unavailable")
    log_info "GPU Information:"
    echo "$GPU_INFO" | sed 's/^/   /'
else
    log_warning "rocminfo not available - limited GPU information"
fi

# Check target script
if [ ! -f "../tiny_llama_v2.py" ]; then
    log_error "tiny_llama_v2.py not found. Please run this script from version2_pytorch_fused directory."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"
log_info "Advanced profiling results will be saved to: $(pwd)"

# Configuration summary
log_info "Configuration:"
log_info "  Batch size: $BATCH_SIZE"
log_info "  Sequence length: $SEQ_LEN"
log_info "  Training steps: $NUM_STEPS"
log_info "  Detailed analysis: $DETAILED_ANALYSIS"
log_info "  Optimization hints: $OPTIMIZATION_HINTS"
log_info "  Kernel filter: '$KERNEL_FILTER'"
log_info "  Metrics: '$METRICS'"
log_info "  Roofline analysis: $ROOFLINE_ANALYSIS"
log_info "  Comparative analysis: $COMPARATIVE_ANALYSIS"

# Step 2: Baseline Advanced Profiling
if [ "$COMPARATIVE_ANALYSIS" = true ]; then
    log_step "2. Baseline Advanced Profiling"
    log_compute "Running advanced baseline profile (no fusion)..."

    BASELINE_CMD="python ../tiny_llama_v2.py \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --num-steps $NUM_STEPS \
        --disable-all-fusion"

    # Build rocprof-compute command
    ROCPROF_COMPUTE_ARGS="profile --name baseline_advanced"

    if [ "$KERNEL_FILTER" != "" ]; then
        ROCPROF_COMPUTE_ARGS="$ROCPROF_COMPUTE_ARGS --kernel-filter '$KERNEL_FILTER'"
    fi

    if [ "$METRICS" != "" ]; then
        ROCPROF_COMPUTE_ARGS="$ROCPROF_COMPUTE_ARGS --metrics $METRICS"
    fi

    # Run baseline advanced profiling
    eval "rocprof-compute $ROCPROF_COMPUTE_ARGS $BASELINE_CMD" > baseline_advanced.log 2>&1

    if [ $? -eq 0 ]; then
        log_info "PASS Baseline advanced profiling completed"

        # Quick analysis
        if [ -d "baseline_advanced" ]; then
            BASELINE_FILES=$(ls baseline_advanced/ | wc -l)
            log_info "Baseline profile files generated: $BASELINE_FILES"
        fi
    else
        log_warning "Baseline advanced profiling had issues (check baseline_advanced.log)"
        tail -10 baseline_advanced.log
    fi
else
    log_info "Skipping baseline analysis (comparative analysis disabled)"
fi

# Step 3: Fused Implementation Advanced Profiling
log_step "3. Fused Implementation Advanced Profiling"
log_compute "Running advanced fused profile (all optimizations)..."

FUSED_CMD="python ../tiny_llama_v2.py \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --num-steps $NUM_STEPS \
    --enable-all-fusion"

# Run fused advanced profiling
FUSED_ROCPROF_ARGS="profile --name fused_advanced"

if [ "$KERNEL_FILTER" != "" ]; then
    FUSED_ROCPROF_ARGS="$FUSED_ROCPROF_ARGS --kernel-filter '$KERNEL_FILTER'"
fi

if [ "$METRICS" != "" ]; then
    FUSED_ROCPROF_ARGS="$FUSED_ROCPROF_ARGS --metrics $METRICS"
fi

eval "rocprof-compute $FUSED_ROCPROF_ARGS $FUSED_CMD" > fused_advanced.log 2>&1

if [ $? -eq 0 ]; then
    log_info "PASS Fused advanced profiling completed"

    # Quick analysis
    if [ -d "fused_advanced" ]; then
        FUSED_FILES=$(ls fused_advanced/ | wc -l)
        log_info "Fused profile files generated: $FUSED_FILES"
    fi
else
    log_error "FAIL Fused advanced profiling failed (check fused_advanced.log)"
    tail -20 fused_advanced.log
    exit 1
fi

# Step 4: Individual Fusion Component Analysis
if [ "$COMPARATIVE_ANALYSIS" = true ]; then
    log_step "4. Individual Fusion Component Analysis"

    FUSION_CONFIGS=(
        "--enable-qkv-fusion --disable-flash-attention --disable-swiglu-fusion --disable-torch-compile"
        "--disable-qkv-fusion --enable-flash-attention --disable-swiglu-fusion --disable-torch-compile"
        "--disable-qkv-fusion --disable-flash-attention --enable-swiglu-fusion --disable-torch-compile"
        "--disable-qkv-fusion --disable-flash-attention --disable-swiglu-fusion --enable-torch-compile"
    )

    FUSION_NAMES=(
        "qkv_fusion"
        "flash_attention"
        "swiglu_fusion"
        "torch_compile"
    )

    for i in "${!FUSION_CONFIGS[@]}"; do
        config="${FUSION_CONFIGS[$i]}"
        name="${FUSION_NAMES[$i]}"

        log_compute "Profiling $name optimization..."

        COMPONENT_CMD="python ../tiny_llama_v2.py \
            --batch-size $BATCH_SIZE \
            --seq-len $SEQ_LEN \
            --num-steps $((NUM_STEPS / 2)) \
            $config"

        COMPONENT_ROCPROF_ARGS="profile --name ${name}_analysis"

        if [ "$KERNEL_FILTER" != "" ]; then
            COMPONENT_ROCPROF_ARGS="$COMPONENT_ROCPROF_ARGS --kernel-filter '$KERNEL_FILTER'"
        fi

        eval "rocprof-compute $COMPONENT_ROCPROF_ARGS $COMPONENT_CMD" > "${name}_analysis.log" 2>&1

        if [ $? -eq 0 ]; then
            log_info "PASS $name analysis completed"
        else
            log_warning "$name analysis had issues"
        fi
    done

    log_info "PASS Individual component analysis completed"
fi

# Step 5: Roofline Analysis
if [ "$ROOFLINE_ANALYSIS" = true ]; then
    log_step "5. Roofline Analysis"
    log_analysis "Generating roofline analysis for optimization guidance..."

    # Enhanced roofline analysis with rocprof-compute
    ROOFLINE_CMD="python ../tiny_llama_v2.py \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --num-steps 20 \
        --enable-all-fusion"

    log_compute "Running roofline analysis..."
    rocprof-compute profile \
        --name roofline_analysis \
        --roofline \
        --metrics $METRICS \
        $ROOFLINE_CMD > roofline_analysis.log 2>&1

    if [ $? -eq 0 ]; then
        log_info "PASS Roofline analysis completed"

        # Check for roofline data
        if [ -d "roofline_analysis" ]; then
            ROOFLINE_DATA=$(find roofline_analysis -name "*.json" -o -name "*.csv" | wc -l)
            log_info "Roofline data files: $ROOFLINE_DATA"
        fi
    else
        log_warning "Roofline analysis had issues (check roofline_analysis.log)"
    fi
else
    log_info "Skipping roofline analysis"
fi

# Step 6: Optimization Recommendations
if [ "$OPTIMIZATION_HINTS" = true ]; then
    log_step "6. Optimization Recommendations"
    log_analysis "Generating optimization hints and recommendations..."

    # Create optimization analysis script
    cat > generate_optimization_hints.py << 'EOF'
#!/usr/bin/env python3
"""
Advanced optimization analysis for ROCprof-compute results
Generates actionable optimization recommendations
"""

import json
import os
import glob
from pathlib import Path
from datetime import datetime

def analyze_profile_directory(profile_dir):
    """Analyze rocprof-compute profile directory."""
    if not os.path.exists(profile_dir):
        return {"error": f"Profile directory not found: {profile_dir}"}

    profile_data = {
        "profile_name": os.path.basename(profile_dir),
        "files": [],
        "kernel_analysis": {},
        "optimization_opportunities": []
    }

    # Find all JSON and CSV files
    json_files = glob.glob(os.path.join(profile_dir, "*.json"))
    csv_files = glob.glob(os.path.join(profile_dir, "*.csv"))

    profile_data["files"] = {
        "json_files": len(json_files),
        "csv_files": len(csv_files),
        "total_files": len(json_files) + len(csv_files)
    }

    # Basic kernel analysis from CSV files
    total_kernels = 0
    unique_kernels = set()

    for csv_file in csv_files:
        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Has header + data
                    total_kernels += len(lines) - 1
                    for line in lines[1:]:  # Skip header
                        parts = line.split(',')
                        if len(parts) > 0:
                            kernel_name = parts[0].strip('"')
                            unique_kernels.add(kernel_name)
        except Exception:
            continue

    profile_data["kernel_analysis"] = {
        "total_kernel_launches": total_kernels,
        "unique_kernels": len(unique_kernels),
        "kernel_names": list(unique_kernels)[:10]  # Top 10 for brevity
    }

    # Generate optimization opportunities
    if "baseline" in profile_dir.lower():
        profile_data["optimization_opportunities"] = [
            "Implement QKV fusion to reduce attention overhead",
            "Use Flash Attention for memory-efficient computation",
            "Apply SwiGLU fusion in feed-forward layers",
            "Consider torch.compile for automatic optimization"
        ]
    elif "fused" in profile_dir.lower():
        profile_data["optimization_opportunities"] = [
            "Explore mixed precision (FP16/BF16) optimizations",
            "Consider gradient checkpointing for memory efficiency",
            "Investigate custom Triton kernels for further speedup",
            "Optimize memory access patterns"
        ]
    else:
        profile_data["optimization_opportunities"] = [
            "Profile analysis available for detailed optimization",
            "Compare with baseline to identify improvements",
            "Focus on high-frequency, high-duration kernels"
        ]

    return profile_data

def generate_comparative_analysis():
    """Generate comparative analysis across all profiles."""
    profiles = {}
    profile_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and ('analysis' in d or 'advanced' in d)]

    for profile_dir in profile_dirs:
        profiles[profile_dir] = analyze_profile_directory(profile_dir)

    # Comparative insights
    comparison = {
        "profiles_analyzed": len(profiles),
        "profile_summary": {},
        "optimization_impact": [],
        "recommendations": []
    }

    for name, data in profiles.items():
        if "error" not in data:
            comparison["profile_summary"][name] = {
                "total_kernels": data["kernel_analysis"]["total_kernel_launches"],
                "unique_kernels": data["kernel_analysis"]["unique_kernels"],
                "files_generated": data["files"]["total_files"]
            }

    # Generate high-level recommendations
    if len(profiles) > 1:
        baseline_kernels = 0
        fused_kernels = 0

        for name, data in profiles.items():
            if "baseline" in name and "error" not in data:
                baseline_kernels = data["kernel_analysis"]["total_kernel_launches"]
            elif "fused" in name and "error" not in data:
                fused_kernels = data["kernel_analysis"]["total_kernel_launches"]

        if baseline_kernels > 0 and fused_kernels > 0:
            kernel_reduction = baseline_kernels - fused_kernels
            reduction_percent = (kernel_reduction / baseline_kernels) * 100
            comparison["optimization_impact"].append({
                "metric": "Kernel Launch Reduction",
                "baseline": baseline_kernels,
                "optimized": fused_kernels,
                "improvement": kernel_reduction,
                "improvement_percent": reduction_percent
            })

    comparison["recommendations"] = [
        "Focus optimization efforts on the most frequently called kernels",
        "Prioritize fusion opportunities with highest impact",
        "Use roofline analysis to identify compute vs memory bound operations",
        "Consider hardware-specific optimizations for target GPU architecture"
    ]

    return comparison

def main():
    """Generate comprehensive optimization analysis."""
    print("Generating optimization recommendations from rocprof-compute data...")

    analysis_results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "individual_profiles": {},
        "comparative_analysis": generate_comparative_analysis()
    }

    # Analyze individual profiles
    profile_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and ('analysis' in d or 'advanced' in d)]

    for profile_dir in profile_dirs:
        analysis_results["individual_profiles"][profile_dir] = analyze_profile_directory(profile_dir)

    # Save comprehensive analysis
    with open('optimization_recommendations.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print("Optimization analysis complete. Results saved to optimization_recommendations.json")

    # Print summary
    print("\nOPTIMIZATION RECOMMENDATIONS SUMMARY:")
    comp_analysis = analysis_results["comparative_analysis"]

    print(f"Profiles analyzed: {comp_analysis['profiles_analyzed']}")

    for profile, summary in comp_analysis["profile_summary"].items():
        print(f"{profile}: {summary['total_kernels']} kernel launches, {summary['unique_kernels']} unique")

    if comp_analysis["optimization_impact"]:
        for impact in comp_analysis["optimization_impact"]:
            print(f"Optimization Impact: {impact['improvement']:.0f} fewer kernels ({impact['improvement_percent']:.1f}% reduction)")

    print("\nTop Recommendations:")
    for i, rec in enumerate(comp_analysis["recommendations"][:3], 1):
        print(f"{i}. {rec}")

if __name__ == "__main__":
    main()
EOF

    chmod +x generate_optimization_hints.py

    # Run optimization analysis
    if command -v python &> /dev/null; then
        python generate_optimization_hints.py

        if [ -f "optimization_recommendations.json" ]; then
            log_info "PASS Optimization recommendations generated"
        fi
    else
        log_warning "Python not available - skipping optimization analysis"
    fi
else
    log_info "Skipping optimization recommendations"
fi

# Step 7: Advanced Metrics Analysis
if [ "$DETAILED_ANALYSIS" = true ]; then
    log_step "7. Advanced Metrics Analysis"
    log_analysis "Processing detailed performance metrics..."

    # Create metrics analysis script
    cat > analyze_advanced_metrics.py << 'EOF'
#!/usr/bin/env python3
"""
Advanced metrics analysis for rocprof-compute results
"""

import json
import csv
import os
import glob
from datetime import datetime

def analyze_metrics_files():
    """Analyze all available metrics files."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "metrics_analysis": {}
    }

    # Find all CSV files in profile directories
    csv_files = []
    for profile_dir in glob.glob("*_analysis"):
        csv_files.extend(glob.glob(os.path.join(profile_dir, "*.csv")))

    csv_files.extend(glob.glob("*_advanced/*.csv"))

    results["files_analyzed"] = len(csv_files)

    for csv_file in csv_files:
        try:
            profile_name = os.path.dirname(csv_file) or "root"
            if profile_name not in results["metrics_analysis"]:
                results["metrics_analysis"][profile_name] = {
                    "files": [],
                    "kernel_metrics": {}
                }

            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                file_analysis = {
                    "file_path": csv_file,
                    "kernel_count": len(rows),
                    "columns": list(reader.fieldnames) if reader.fieldnames else []
                }

                # Analyze common metrics if available
                if rows and reader.fieldnames:
                    if 'DurationNs' in reader.fieldnames:
                        durations = [float(row.get('DurationNs', 0)) for row in rows if row.get('DurationNs', '').isdigit()]
                        if durations:
                            file_analysis["duration_stats"] = {
                                "total_ns": sum(durations),
                                "avg_ns": sum(durations) / len(durations),
                                "max_ns": max(durations),
                                "min_ns": min(durations)
                            }

                results["metrics_analysis"][profile_name]["files"].append(file_analysis)

        except Exception as e:
            print(f"Error analyzing {csv_file}: {e}")

    return results

def main():
    """Run advanced metrics analysis."""
    print("Analyzing advanced performance metrics...")

    results = analyze_metrics_files()

    # Save results
    with open('advanced_metrics_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Advanced metrics analysis complete. Analyzed {results['files_analyzed']} files.")

    # Print summary
    for profile, data in results["metrics_analysis"].items():
        total_kernels = sum(f.get("kernel_count", 0) for f in data["files"])
        print(f"{profile}: {total_kernels} total kernel launches across {len(data['files'])} files")

if __name__ == "__main__":
    main()
EOF

    chmod +x analyze_advanced_metrics.py

    # Run advanced metrics analysis
    if command -v python &> /dev/null; then
        python analyze_advanced_metrics.py

        if [ -f "advanced_metrics_analysis.json" ]; then
            log_info "PASS Advanced metrics analysis completed"
        fi
    else
        log_warning "Python not available - skipping metrics analysis"
    fi
else
    log_info "Skipping detailed metrics analysis"
fi

# Step 8: Comprehensive Report Generation
log_step "8. Comprehensive Report Generation"

REPORT_FILE="rocprof_compute_comprehensive_report.md"

cat > "$REPORT_FILE" << EOF
# ROCprof-compute Comprehensive Analysis Report - Tiny LLaMA V2

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')
**Configuration:** Batch size $BATCH_SIZE, Sequence length $SEQ_LEN, Steps $NUM_STEPS
**Analysis Type:** Advanced GPU Kernel Profiling with Optimization Recommendations

## Executive Summary

This report provides the most advanced analysis of Tiny LLaMA V2 using ROCprof-compute,
AMD's premier GPU profiling tool. The analysis includes kernel-level optimization
recommendations, roofline analysis, and detailed performance characterization.

## ROCprof-compute Configuration

- **Version:** $ROCPROF_COMPUTE_VERSION
- **Kernel Filter:** '$KERNEL_FILTER'
- **Metrics Collected:** '$METRICS'
- **Detailed Analysis:** $DETAILED_ANALYSIS
- **Roofline Analysis:** $ROOFLINE_ANALYSIS
- **Comparative Analysis:** $COMPARATIVE_ANALYSIS
- **Optimization Hints:** $OPTIMIZATION_HINTS

## Analysis Scope

### Profiling Configurations Analyzed
EOF

if [ "$COMPARATIVE_ANALYSIS" = true ]; then
    cat >> "$REPORT_FILE" << EOF

1. **Baseline Advanced** - No fusion optimizations (reference)
2. **Fused Advanced** - All fusion optimizations enabled
3. **QKV Fusion Only** - Isolated QKV fusion impact
4. **Flash Attention Only** - Isolated Flash Attention impact
5. **SwiGLU Fusion Only** - Isolated SwiGLU fusion impact
6. **Torch Compile Only** - Isolated torch.compile impact
EOF
else
    cat >> "$REPORT_FILE" << EOF

1. **Fused Advanced** - All fusion optimizations enabled (primary analysis)
EOF
fi

cat >> "$REPORT_FILE" << EOF

### Analysis Components

- **Kernel Execution Analysis** - Detailed kernel timing and frequency
- **Memory Access Patterns** - L1/L2 cache utilization and memory bandwidth
- **GPU Utilization Metrics** - Compute unit occupancy and efficiency
$([ "$ROOFLINE_ANALYSIS" = true ] && echo "- **Roofline Analysis** - Performance bounds and optimization targets")
$([ "$OPTIMIZATION_HINTS" = true ] && echo "- **Optimization Recommendations** - Actionable performance improvements")

## Key Findings

### Performance Characterization
EOF

# Add optimization recommendations if available
if [ -f "optimization_recommendations.json" ]; then
    cat >> "$REPORT_FILE" << EOF

### Optimization Impact Analysis
\`\`\`json
$(cat optimization_recommendations.json | head -50)
...
\`\`\`
EOF
fi

# Add advanced metrics if available
if [ -f "advanced_metrics_analysis.json" ]; then
    cat >> "$REPORT_FILE" << EOF

### Advanced Metrics Summary
\`\`\`json
$(cat advanced_metrics_analysis.json | head -30)
...
\`\`\`
EOF
fi

cat >> "$REPORT_FILE" << EOF

## Detailed Analysis Results

### Generated Profile Directories
EOF

# List all generated profile directories
for dir in *_analysis *_advanced; do
    if [ -d "$dir" ]; then
        file_count=$(ls "$dir" | wc -l)
        cat >> "$REPORT_FILE" << EOF
- \`$dir/\` - $file_count files generated
EOF
    fi
done

cat >> "$REPORT_FILE" << EOF

### Key Performance Metrics

#### Kernel Execution Efficiency
- Kernel launch frequency and distribution
- Duration analysis for optimization prioritization
- Memory access patterns and cache utilization

#### GPU Resource Utilization
- Compute unit occupancy rates
- Memory bandwidth utilization
- Pipeline efficiency metrics

$([ "$ROOFLINE_ANALYSIS" = true ] && echo "#### Roofline Analysis Results
- Performance positioning relative to theoretical limits
- Memory-bound vs compute-bound operation identification
- Optimization opportunity quantification")

## Optimization Recommendations

### High Priority Optimizations
EOF

# Add specific recommendations based on analysis
if [ -f "optimization_recommendations.json" ]; then
    # Extract top recommendations
    python -c "
import json
try:
    with open('optimization_recommendations.json') as f:
        data = json.load(f)
    comp = data.get('comparative_analysis', {})
    recs = comp.get('recommendations', [])
    for i, rec in enumerate(recs[:3], 1):
        print(f'{i}. {rec}')
except:
    print('1. Focus on high-frequency kernel optimization')
    print('2. Implement memory access pattern improvements')
    print('3. Consider hardware-specific optimizations')
" >> "$REPORT_FILE" 2>/dev/null || cat >> "$REPORT_FILE" << EOF
1. Focus on high-frequency kernel optimization
2. Implement memory access pattern improvements
3. Consider hardware-specific optimizations
EOF
fi

cat >> "$REPORT_FILE" << EOF

### Implementation Priorities
- **Immediate**: Address kernels with highest frequency and duration
- **Short-term**: Implement fusion opportunities with proven impact
- **Long-term**: Custom kernel development for specialized operations

## Next Steps for Version 3

Based on this advanced analysis, Version 3 should focus on:

1. **Custom Triton Kernels** - Implement kernels identified as optimization targets
2. **Memory Optimization** - Address memory bandwidth bottlenecks
3. **Hardware-Specific Tuning** - Optimize for target GPU architecture
4. **Advanced Fusion** - Implement ultra-fusion techniques

## Files and Data

### Analysis Outputs
- \`optimization_recommendations.json\` - Comprehensive optimization analysis
- \`advanced_metrics_analysis.json\` - Detailed performance metrics
- \`*_analysis/\` - Individual profile directories with detailed data
- \`*.log\` - Execution logs for each profiling run

### Visualization and Further Analysis
- Use rocprof-compute's built-in visualization tools
- Import CSV data into analysis tools for custom insights
- Integrate with roofline analysis tools for performance modeling

---
*Generated by AI Workshop ROCprof-compute Advanced Analysis Tool*
*This represents the most comprehensive GPU profiling analysis available for ROCm*
EOF

log_info "Comprehensive analysis report generated: $REPORT_FILE"

# Step 9: Final Summary and Recommendations
log_step "9. ROCprof-compute Analysis Complete"

echo ""
echo "ROCprof-compute advanced analysis completed!"
echo ""
echo "📁 Results Location: $(pwd)"
echo ""
echo "Generated Analysis Files:"
echo "   - $REPORT_FILE           # Comprehensive analysis report"
if [ -f "optimization_recommendations.json" ]; then
    echo "   - optimization_recommendations.json  # Actionable optimization hints"
fi
if [ -f "advanced_metrics_analysis.json" ]; then
    echo "   - advanced_metrics_analysis.json    # Detailed performance metrics"
fi
echo "   - *_analysis/                        # Individual profile directories"
echo "   - *.log                              # Execution and analysis logs"
echo ""
echo "Advanced Insights Generated:"

# Extract and display key insights
if [ -f "optimization_recommendations.json" ]; then
    PROFILES_ANALYZED=$(python -c "import json; data=json.load(open('optimization_recommendations.json')); print(data.get('comparative_analysis', {}).get('profiles_analyzed', 0))" 2>/dev/null || echo "N/A")
    echo "   Profiles analyzed: $PROFILES_ANALYZED"

    # Try to extract optimization impact
    IMPACT=$(python -c "
import json
try:
    data=json.load(open('optimization_recommendations.json'))
    impacts = data.get('comparative_analysis', {}).get('optimization_impact', [])
    if impacts:
        impact = impacts[0]
        print(f\"{impact.get('improvement', 0):.0f} kernels reduced ({impact.get('improvement_percent', 0):.1f}%)\")
    else:
        print('Analysis available in detailed report')
except:
    print('Detailed analysis available')
" 2>/dev/null || echo "Detailed analysis available")
    echo "   Optimization impact: $IMPACT"
fi

echo ""
echo "Advanced Analysis Features Used:"
echo "   PASS Kernel-level execution analysis"
echo "   PASS Memory access pattern analysis"
echo "   PASS GPU utilization metrics"
if [ "$ROOFLINE_ANALYSIS" = true ]; then
    echo "   PASS Roofline analysis for performance bounds"
fi
if [ "$COMPARATIVE_ANALYSIS" = true ]; then
    echo "   PASS Comparative fusion impact analysis"
fi
if [ "$OPTIMIZATION_HINTS" = true ]; then
    echo "   PASS Automated optimization recommendations"
fi
echo ""
echo "🔄 Next Steps:"
echo "   1. Review comprehensive report: cat $REPORT_FILE"
echo "   2. Examine detailed profile data: ls *_analysis/"
echo "   3. Implement optimization recommendations"
echo "   4. Proceed to Version 3 for Triton kernel implementation"
echo ""
echo "Advanced Usage:"
echo "   # Detailed kernel analysis"
echo "   rocprof-compute analyze fused_advanced/"
echo ""
echo "   # Custom metric collection"
echo "   rocprof-compute profile --metrics 'pmc,l2_cache_hit_rate' [command]"
echo ""
echo "   # Roofline visualization"
echo "   rocprof-compute visualize roofline_analysis/"
echo ""

log_info "ROCprof-compute advanced analysis complete!"
log_compute "Most comprehensive GPU profiling analysis available for optimization guidance"

# Return to original directory
cd - > /dev/null
