#!/bin/bash

# Comprehensive Profiling Suite for Tiny LLaMA V2 with ROCm Tools Integration
# This script orchestrates all profiling tools for complete fusion analysis

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

log_profiler() {
    echo -e "${PURPLE}[PROFILER]${NC} $1"
}

log_analysis() {
    echo -e "${CYAN}[ANALYSIS]${NC} $1"
}

# Default configuration
BATCH_SIZE=8
SEQ_LEN=128
NUM_STEPS=50
PROFILE_DIR="./complete_v2_analysis_$(date +%Y%m%d_%H%M%S)"
ENABLE_PYTORCH_PROFILER=true
ENABLE_DEEPSPEED_FLOPS=true
ENABLE_ROCM_TOOLS=true
ENABLE_ROCPROFV3=true
ENABLE_ROCPROF_SYS=true
ENABLE_ROCPROF_COMPUTE=true
COMPARATIVE_ANALYSIS=true
GENERATE_REPORTS=true
VALIDATE_ENV=true
QUICK_MODE=false

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
        --enable-pytorch-profiler)
            ENABLE_PYTORCH_PROFILER=true
            shift
            ;;
        --disable-pytorch-profiler)
            ENABLE_PYTORCH_PROFILER=false
            shift
            ;;
        --enable-deepspeed-flops)
            ENABLE_DEEPSPEED_FLOPS=true
            shift
            ;;
        --disable-deepspeed-flops)
            ENABLE_DEEPSPEED_FLOPS=false
            shift
            ;;
        --enable-rocm-tools)
            ENABLE_ROCM_TOOLS=true
            shift
            ;;
        --disable-rocm-tools)
            ENABLE_ROCM_TOOLS=false
            ENABLE_ROCPROFV3=false
            ENABLE_ROCPROF_SYS=false
            ENABLE_ROCPROF_COMPUTE=false
            shift
            ;;
        --enable-rocprofv3)
            ENABLE_ROCPROFV3=true
            shift
            ;;
        --disable-rocprofv3)
            ENABLE_ROCPROFV3=false
            shift
            ;;
        --enable-rocprof-sys)
            ENABLE_ROCPROF_SYS=true
            shift
            ;;
        --disable-rocprof-sys)
            ENABLE_ROCPROF_SYS=false
            shift
            ;;
        --enable-rocprof-compute)
            ENABLE_ROCPROF_COMPUTE=true
            shift
            ;;
        --disable-rocprof-compute)
            ENABLE_ROCPROF_COMPUTE=false
            shift
            ;;
        --no-comparative)
            COMPARATIVE_ANALYSIS=false
            shift
            ;;
        --no-reports)
            GENERATE_REPORTS=false
            shift
            ;;
        --skip-validation)
            VALIDATE_ENV=false
            shift
            ;;
        --quick-mode)
            QUICK_MODE=true
            NUM_STEPS=20
            shift
            ;;
        --help|-h)
            echo "Comprehensive Profiling Suite for Tiny LLaMA V2"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Configuration Options:"
            echo "  --batch-size SIZE              Batch size (default: 8)"
            echo "  --seq-len LENGTH               Sequence length (default: 128)"
            echo "  --num-steps STEPS              Training steps (default: 50)"
            echo "  --profile-dir DIR              Profile output directory"
            echo ""
            echo "Profiler Control:"
            echo "  --enable-pytorch-profiler      Enable PyTorch profiler (default)"
            echo "  --disable-pytorch-profiler     Disable PyTorch profiler"
            echo "  --enable-deepspeed-flops       Enable DeepSpeed FLOPS profiler (default)"
            echo "  --disable-deepspeed-flops      Disable DeepSpeed FLOPS profiler"
            echo "  --enable-rocm-tools            Enable all ROCm tools (default)"
            echo "  --disable-rocm-tools           Disable all ROCm tools"
            echo "  --enable-rocprofv3             Enable ROCprofv3 specifically"
            echo "  --disable-rocprofv3            Disable ROCprofv3"
            echo "  --enable-rocprof-sys           Enable ROCprof-sys specifically"
            echo "  --disable-rocprof-sys          Disable ROCprof-sys"
            echo "  --enable-rocprof-compute       Enable ROCprof-compute specifically"
            echo "  --disable-rocprof-compute      Disable ROCprof-compute"
            echo ""
            echo "Analysis Options:"
            echo "  --no-comparative               Skip comparative fusion analysis"
            echo "  --no-reports                   Skip report generation"
            echo "  --skip-validation              Skip environment validation"
            echo "  --quick-mode                   Fast profiling (reduces steps to 20)"
            echo ""
            echo "  --help, -h                     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                           # Complete analysis with all tools"
            echo "  $0 --batch-size 16 --seq-len 256            # Custom model configuration"
            echo "  $0 --disable-rocm-tools --quick-mode        # PyTorch/DeepSpeed only, fast mode"
            echo "  $0 --enable-rocprof-compute --no-comparative # ROCprof-compute only"
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
echo "CASTILLE AI WORKSHOP - COMPREHENSIVE PROFILING SUITE V2"
echo "     Fusion Optimization Analysis with Complete ROCm Tools Integration"
echo "=" * 80
echo ""

log_info "Profiling Configuration:"
log_info "  Batch size: $BATCH_SIZE"
log_info "  Sequence length: $SEQ_LEN"
log_info "  Training steps: $NUM_STEPS"
log_info "  Profile directory: $PROFILE_DIR"
log_info "  Quick mode: $QUICK_MODE"

log_info ""
log_info "Profiling Tools Enabled:"
log_info "  PyTorch Profiler: $ENABLE_PYTORCH_PROFILER"
log_info "  DeepSpeed FLOPS: $ENABLE_DEEPSPEED_FLOPS"
log_info "  ROCm Tools Suite: $ENABLE_ROCM_TOOLS"
log_info "    - ROCprofv3: $ENABLE_ROCPROFV3"
log_info "    - ROCprof-sys: $ENABLE_ROCPROF_SYS"
log_info "    - ROCprof-compute: $ENABLE_ROCPROF_COMPUTE"

log_info ""
log_info "Analysis Options:"
log_info "  Comparative analysis: $COMPARATIVE_ANALYSIS"
log_info "  Generate reports: $GENERATE_REPORTS"
log_info "  Environment validation: $VALIDATE_ENV"

# Create profile directory
mkdir -p "$PROFILE_DIR"
cd "$PROFILE_DIR"
log_info "Created comprehensive profile directory: $(pwd)"

# Environment validation
if [ "$VALIDATE_ENV" = true ]; then
    log_step "1. Environment Validation"

    # Check Python environment
    if ! command -v python &> /dev/null; then
        log_error "Python not found. Please ensure Python is installed and in PATH."
        exit 1
    fi

    # Check if we're in the right directory
    if [ ! -f "../tiny_llama_v2.py" ]; then
        log_error "tiny_llama_v2.py not found. Please run this script from version2_pytorch_fused directory."
        exit 1
    fi

    # Quick environment test
    log_info "Running V2 environment validation..."
    python ../tiny_llama_v2.py --validate-setup --batch-size 2 --num-steps 2 > validation.log 2>&1

    if [ $? -eq 0 ]; then
        log_info "PASS V2 environment validation passed"
        rm -f validation.log
    else
        log_error "FAIL V2 environment validation failed. Check validation.log for details."
        cat validation.log
        exit 1
    fi

    # Check ROCm tools availability
    if [ "$ENABLE_ROCM_TOOLS" = true ]; then
        log_info "Validating ROCm tools availability..."

        ROCM_TOOLS_STATUS=""
        if [ "$ENABLE_ROCPROFV3" = true ]; then
            if command -v rocprof &> /dev/null; then
                ROCM_TOOLS_STATUS="$ROCM_TOOLS_STATUS ROCprofv3:PASS"
            else
                ROCM_TOOLS_STATUS="$ROCM_TOOLS_STATUS ROCprofv3:FAIL"
                log_warning "ROCprofv3 not available"
            fi
        fi

        if [ "$ENABLE_ROCPROF_SYS" = true ]; then
            if command -v rocprof-sys &> /dev/null; then
                ROCM_TOOLS_STATUS="$ROCM_TOOLS_STATUS ROCprof-sys:PASS"
            else
                ROCM_TOOLS_STATUS="$ROCM_TOOLS_STATUS ROCprof-sys:FAIL"
                log_warning "ROCprof-sys not available"
            fi
        fi

        if [ "$ENABLE_ROCPROF_COMPUTE" = true ]; then
            if command -v rocprof-compute &> /dev/null; then
                ROCM_TOOLS_STATUS="$ROCM_TOOLS_STATUS ROCprof-compute:PASS"
            else
                ROCM_TOOLS_STATUS="$ROCM_TOOLS_STATUS ROCprof-compute:FAIL"
                log_warning "ROCprof-compute not available"
            fi
        fi

        log_info "ROCm Tools Status:$ROCM_TOOLS_STATUS"
    fi

else
    log_warning "Skipping environment validation"
fi

# Step 2: Baseline Performance Run (V2 vs V1 Comparison)
log_step "2. Baseline Performance Establishment"

if [ "$COMPARATIVE_ANALYSIS" = true ]; then
    log_info "Running V2 baseline performance measurement..."

    # Run V2 with all fusion disabled (equivalent to V1 performance)
    python ../tiny_llama_v2.py \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --num-steps $NUM_STEPS \
        --disable-all-fusion \
        > v2_baseline_performance.log 2>&1

    if [ $? -eq 0 ]; then
        log_info "PASS V2 baseline performance established"

        # Extract key metrics
        BASELINE_SPEED=$(grep "samples/sec" v2_baseline_performance.log | tail -1 | grep -o '[0-9.]*\s*samples/sec' | grep -o '[0-9.]*' || echo "N/A")
        BASELINE_MEMORY=$(grep "Memory:" v2_baseline_performance.log | tail -1 | grep -o '[0-9.]*\s*MB' | grep -o '[0-9.]*' || echo "N/A")

        log_info "   Baseline speed: ${BASELINE_SPEED} samples/sec"
        log_info "   Baseline memory: ${BASELINE_MEMORY} MB"
    else
        log_error "FAIL V2 baseline performance measurement failed"
        tail -20 v2_baseline_performance.log
        exit 1
    fi

    # Run V2 with all fusion enabled for comparison
    log_info "Running V2 fused performance measurement..."

    python ../tiny_llama_v2.py \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --num-steps $NUM_STEPS \
        --enable-all-fusion \
        > v2_fused_performance.log 2>&1

    if [ $? -eq 0 ]; then
        log_info "PASS V2 fused performance established"

        # Extract key metrics and calculate improvement
        FUSED_SPEED=$(grep "samples/sec" v2_fused_performance.log | tail -1 | grep -o '[0-9.]*\s*samples/sec' | grep -o '[0-9.]*' || echo "N/A")
        FUSED_MEMORY=$(grep "Memory:" v2_fused_performance.log | tail -1 | grep -o '[0-9.]*\s*MB' | grep -o '[0-9.]*' || echo "N/A")

        if [ "$BASELINE_SPEED" != "N/A" ] && [ "$FUSED_SPEED" != "N/A" ]; then
            SPEEDUP=$(python -c "print(f'{float('$FUSED_SPEED')/float('$BASELINE_SPEED'):.2f}x')" 2>/dev/null || echo "N/A")
            log_info "   Fused speed: ${FUSED_SPEED} samples/sec"
            log_info "   Fusion speedup: ${SPEEDUP}"
        fi

        log_info "   Fused memory: ${FUSED_MEMORY} MB"
    else
        log_warning "V2 fused performance measurement had issues"
    fi
else
    log_info "Skipping baseline performance comparison"
fi

# Step 3: PyTorch Profiler Analysis
if [ "$ENABLE_PYTORCH_PROFILER" = true ]; then
    log_step "3. PyTorch Profiler Analysis"
    log_profiler "Running enhanced PyTorch profiling with fusion analysis..."

    PYTORCH_DIR="pytorch_v2_profiling"
    mkdir -p "$PYTORCH_DIR"

    python ../run_pytorch_profiler.py \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --num-steps $(($NUM_STEPS / 2)) \
        --profile-dir "$PYTORCH_DIR" \
        --include-memory \
        --include-shapes \
        --generate-report \
        > pytorch_v2_profiling.log 2>&1

    if [ $? -eq 0 ]; then
        log_info "PASS PyTorch profiling completed"
        log_info "📁 Results saved to: $PYTORCH_DIR"
    else
        log_warning "PyTorch profiling had issues (check pytorch_v2_profiling.log)"
        tail -10 pytorch_v2_profiling.log
    fi
else
    log_info "Skipping PyTorch profiler analysis"
fi

# Step 4: DeepSpeed FLOPS Analysis
if [ "$ENABLE_DEEPSPEED_FLOPS" = true ]; then
    log_step "4. DeepSpeed FLOPS Analysis"
    log_profiler "Running FLOPS analysis with fusion impact measurement..."

    FLOPS_DIR="deepspeed_v2_flops"
    mkdir -p "$FLOPS_DIR"

    python ../run_deepspeed_flops.py \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --num-steps $(($NUM_STEPS / 2)) \
        --output-dir "$FLOPS_DIR" \
        --detailed-analysis \
        --computational-intensity \
        --generate-roofline \
        > deepspeed_v2_flops.log 2>&1

    if [ $? -eq 0 ]; then
        log_info "PASS FLOPS analysis completed"
        log_info "📁 Results saved to: $FLOPS_DIR"

        # Extract key FLOPS metrics
        if [ -f "$FLOPS_DIR/flops_profile.json" ]; then
            MFU=$(python -c "import json; data=json.load(open('$FLOPS_DIR/flops_profile.json')); print(f\"{data['efficiency_metrics']['mfu_percent']:.1f}%\")" 2>/dev/null || echo "N/A")
            THROUGHPUT=$(python -c "import json; data=json.load(open('$FLOPS_DIR/flops_profile.json')); print(f\"{data['performance_metrics']['throughput_samples_per_sec']:.1f}\")" 2>/dev/null || echo "N/A")
            log_info "Model FLOPS Utilization (V2): $MFU"
            log_info "FLOPS Analysis Throughput: $THROUGHPUT samples/sec"
        fi
    else
        log_warning "FLOPS analysis had issues (check deepspeed_v2_flops.log)"
        tail -10 deepspeed_v2_flops.log
    fi
else
    log_info "Skipping DeepSpeed FLOPS analysis"
fi

# Step 5: ROCprofv3 Kernel Analysis
if [ "$ENABLE_ROCPROFV3" = true ] && [ "$ENABLE_ROCM_TOOLS" = true ]; then
    log_step "5. ROCprofv3 Kernel Analysis"
    log_profiler "Running ROCprofv3 kernel-level profiling..."

    ROCPROFV3_DIR="rocprofv3_v2_analysis"

    bash ../run_rocprofv3.sh \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --num-steps $(($NUM_STEPS / 2)) \
        --output-dir "$ROCPROFV3_DIR" \
        --profile-kernels \
        --detailed-metrics \
        > rocprofv3_v2.log 2>&1

    if [ $? -eq 0 ]; then
        log_info "PASS ROCprofv3 analysis completed"
        log_info "📁 Results saved to: $ROCPROFV3_DIR"

        # Extract kernel reduction metrics if available
        if [ -f "$ROCPROFV3_DIR/rocprof_analysis.json" ]; then
            log_info "ROCprofv3 kernel analysis data available"
        fi
    else
        log_warning "ROCprofv3 analysis had issues (check rocprofv3_v2.log)"
        tail -10 rocprofv3_v2.log
    fi
else
    log_info "Skipping ROCprofv3 analysis"
fi

# Step 6: ROCprof-sys System Analysis
if [ "$ENABLE_ROCPROF_SYS" = true ] && [ "$ENABLE_ROCM_TOOLS" = true ]; then
    log_step "6. ROCprof-sys System Analysis"
    log_profiler "Running ROCprof-sys system-wide profiling..."

    ROCPROF_SYS_DIR="rocprof_sys_v2_analysis"
    DURATION=90  # Adjusted for V2 analysis

    bash ../run_rocprof_sys.sh \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --num-steps $(($NUM_STEPS / 2)) \
        --duration $DURATION \
        --output-dir "$ROCPROF_SYS_DIR" \
        --monitor-cpu \
        --monitor-gpu \
        --monitor-memory \
        > rocprof_sys_v2.log 2>&1

    if [ $? -eq 0 ]; then
        log_info "PASS ROCprof-sys analysis completed"
        log_info "📁 Results saved to: $ROCPROF_SYS_DIR"

        # Extract system utilization metrics if available
        if [ -f "$ROCPROF_SYS_DIR/system_analysis.json" ]; then
            CPU_UTIL=$(python -c "import json; data=json.load(open('$ROCPROF_SYS_DIR/system_analysis.json')); print(f\"{data.get('system_resources', {}).get('resource_analysis', {}).get('cpu_utilization', {}).get('mean', 0):.1f}%\")" 2>/dev/null || echo "N/A")
            GPU_UTIL=$(python -c "import json; data=json.load(open('$ROCPROF_SYS_DIR/system_analysis.json')); print(f\"{data.get('system_resources', {}).get('gpu_analysis', {}).get('utilization', {}).get('mean', 0):.1f}%\")" 2>/dev/null || echo "N/A")
            log_info "Average CPU Utilization: $CPU_UTIL"
            log_info "Average GPU Utilization: $GPU_UTIL"
        fi
    else
        log_warning "ROCprof-sys analysis had issues (check rocprof_sys_v2.log)"
        tail -10 rocprof_sys_v2.log
    fi
else
    log_info "Skipping ROCprof-sys analysis"
fi

# Step 7: ROCprof-compute Advanced Analysis
if [ "$ENABLE_ROCPROF_COMPUTE" = true ] && [ "$ENABLE_ROCM_TOOLS" = true ]; then
    log_step "7. ROCprof-compute Advanced Analysis"
    log_profiler "Running ROCprof-compute advanced kernel profiling..."

    ROCPROF_COMPUTE_DIR="rocprof_compute_v2_analysis"

    bash ../run_rocprof_compute.sh \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --num-steps $(($NUM_STEPS / 2)) \
        --output-dir "$ROCPROF_COMPUTE_DIR" \
        --detailed-analysis \
        --optimization-hints \
        --roofline-analysis \
        --comparative-analysis \
        > rocprof_compute_v2.log 2>&1

    if [ $? -eq 0 ]; then
        log_info "PASS ROCprof-compute analysis completed"
        log_info "📁 Results saved to: $ROCPROF_COMPUTE_DIR"

        # Extract optimization recommendations if available
        if [ -f "$ROCPROF_COMPUTE_DIR/optimization_recommendations.json" ]; then
            PROFILES_ANALYZED=$(python -c "import json; data=json.load(open('$ROCPROF_COMPUTE_DIR/optimization_recommendations.json')); print(data.get('comparative_analysis', {}).get('profiles_analyzed', 0))" 2>/dev/null || echo "N/A")
            log_info "Profiles analyzed by ROCprof-compute: $PROFILES_ANALYZED"
        fi
    else
        log_warning "ROCprof-compute analysis had issues (check rocprof_compute_v2.log)"
        tail -10 rocprof_compute_v2.log
    fi
else
    log_info "Skipping ROCprof-compute analysis"
fi

# Step 8: Cross-Tool Analysis and Report Generation
if [ "$GENERATE_REPORTS" = true ]; then
    log_step "8. Cross-Tool Analysis and Comprehensive Reporting"
    log_analysis "Generating comprehensive V2 fusion analysis report..."

    # Create comprehensive analysis script
    cat > generate_v2_comprehensive_report.py << 'EOF'
#!/usr/bin/env python3
"""
Comprehensive V2 Analysis Report Generator
Integrates data from all profiling tools for unified insights
"""

import json
import os
import glob
from datetime import datetime
from pathlib import Path

def load_json_safe(filepath):
    """Safely load JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to load {filepath}: {str(e)}"}

def extract_performance_metrics():
    """Extract performance metrics from all available sources."""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "v2_performance": {},
        "pytorch_profiling": {},
        "deepspeed_flops": {},
        "rocm_analysis": {},
        "comparative_analysis": {}
    }

    # V2 Performance metrics
    if os.path.exists("v2_baseline_performance.log"):
        with open("v2_baseline_performance.log", 'r') as f:
            content = f.read()
            # Extract baseline metrics
            lines = content.split('\n')
            for line in lines:
                if "samples/sec" in line:
                    try:
                        speed = line.split("samples/sec")[0].split()[-1]
                        metrics["v2_performance"]["baseline_speed"] = float(speed)
                    except:
                        pass

    if os.path.exists("v2_fused_performance.log"):
        with open("v2_fused_performance.log", 'r') as f:
            content = f.read()
            # Extract fused metrics
            lines = content.split('\n')
            for line in lines:
                if "samples/sec" in line:
                    try:
                        speed = line.split("samples/sec")[0].split()[-1]
                        metrics["v2_performance"]["fused_speed"] = float(speed)
                    except:
                        pass

    # Calculate speedup
    if "baseline_speed" in metrics["v2_performance"] and "fused_speed" in metrics["v2_performance"]:
        baseline = metrics["v2_performance"]["baseline_speed"]
        fused = metrics["v2_performance"]["fused_speed"]
        if baseline > 0:
            metrics["v2_performance"]["fusion_speedup"] = fused / baseline

    # DeepSpeed FLOPS metrics
    flops_file = "deepspeed_v2_flops/flops_profile.json"
    if os.path.exists(flops_file):
        metrics["deepspeed_flops"] = load_json_safe(flops_file)

    # ROCprof-compute optimization recommendations
    rocprof_compute_file = "rocprof_compute_v2_analysis/optimization_recommendations.json"
    if os.path.exists(rocprof_compute_file):
        metrics["rocm_analysis"]["rocprof_compute"] = load_json_safe(rocprof_compute_file)

    # ROCprof-sys system analysis
    rocprof_sys_file = "rocprof_sys_v2_analysis/system_analysis.json"
    if os.path.exists(rocprof_sys_file):
        metrics["rocm_analysis"]["rocprof_sys"] = load_json_safe(rocprof_sys_file)

    # ROCprofv3 kernel analysis
    rocprofv3_file = "rocprofv3_v2_analysis/rocprof_analysis.json"
    if os.path.exists(rocprofv3_file):
        metrics["rocm_analysis"]["rocprofv3"] = load_json_safe(rocprofv3_file)

    return metrics

def generate_executive_summary(metrics):
    """Generate executive summary of all analyses."""
    summary = {
        "fusion_impact": {},
        "key_findings": [],
        "optimization_priorities": [],
        "tool_insights": {}
    }

    # Fusion impact
    v2_perf = metrics.get("v2_performance", {})
    if "fusion_speedup" in v2_perf:
        speedup = v2_perf["fusion_speedup"]
        summary["fusion_impact"]["overall_speedup"] = f"{speedup:.2f}x"

        if speedup >= 2.0:
            summary["key_findings"].append(f"Exceptional fusion performance: {speedup:.2f}x speedup achieved")
        elif speedup >= 1.5:
            summary["key_findings"].append(f"Strong fusion impact: {speedup:.2f}x speedup demonstrates optimization success")
        else:
            summary["key_findings"].append(f"Moderate fusion benefit: {speedup:.2f}x speedup with room for improvement")

    # FLOPS analysis insights
    flops_data = metrics.get("deepspeed_flops", {})
    if "efficiency_metrics" in flops_data and "error" not in flops_data:
        mfu = flops_data["efficiency_metrics"].get("mfu_percent", 0)
        summary["tool_insights"]["flops_utilization"] = f"{mfu:.1f}% MFU"

        if mfu >= 50:
            summary["key_findings"].append(f"High computational efficiency: {mfu:.1f}% Model FLOPS Utilization")
        elif mfu >= 30:
            summary["key_findings"].append(f"Good computational efficiency: {mfu:.1f}% MFU with optimization potential")
        else:
            summary["key_findings"].append(f"Low computational efficiency: {mfu:.1f}% MFU indicates significant optimization opportunities")

    # ROCm tools insights
    rocm_data = metrics.get("rocm_analysis", {})

    # ROCprof-compute recommendations
    rocprof_compute = rocm_data.get("rocprof_compute", {})
    if "comparative_analysis" in rocprof_compute and "error" not in rocprof_compute:
        comp_analysis = rocprof_compute["comparative_analysis"]
        profiles_analyzed = comp_analysis.get("profiles_analyzed", 0)

        if profiles_analyzed > 0:
            summary["tool_insights"]["rocprof_compute"] = f"{profiles_analyzed} profiles analyzed with optimization recommendations"

            recommendations = comp_analysis.get("recommendations", [])
            summary["optimization_priorities"].extend(recommendations[:3])  # Top 3

    # System resource analysis
    rocprof_sys = rocm_data.get("rocprof_sys", {})
    if "system_resources" in rocprof_sys and "error" not in rocprof_sys:
        sys_resources = rocprof_sys["system_resources"]
        if "resource_analysis" in sys_resources:
            cpu_util = sys_resources["resource_analysis"].get("cpu_utilization", {}).get("mean", 0)
            gpu_util_data = sys_resources.get("gpu_analysis", {}).get("utilization", {})
            gpu_util = gpu_util_data.get("mean", 0) if isinstance(gpu_util_data, dict) else 0

            summary["tool_insights"]["system_utilization"] = f"CPU: {cpu_util:.1f}%, GPU: {gpu_util:.1f}%"

    return summary

def main():
    """Generate comprehensive V2 analysis report."""
    print("Generating comprehensive V2 fusion analysis report...")

    # Extract all metrics
    metrics = extract_performance_metrics()

    # Generate executive summary
    exec_summary = generate_executive_summary(metrics)

    # Combine all data
    comprehensive_report = {
        "report_metadata": {
            "generated": datetime.now().isoformat(),
            "version": "Tiny LLaMA V2 Comprehensive Analysis",
            "tools_used": []
        },
        "executive_summary": exec_summary,
        "detailed_metrics": metrics
    }

    # Identify tools used
    tools_used = []
    if os.path.exists("pytorch_v2_profiling"):
        tools_used.append("PyTorch Profiler")
    if os.path.exists("deepspeed_v2_flops"):
        tools_used.append("DeepSpeed FLOPS Profiler")
    if os.path.exists("rocprofv3_v2_analysis"):
        tools_used.append("ROCprofv3")
    if os.path.exists("rocprof_sys_v2_analysis"):
        tools_used.append("ROCprof-sys")
    if os.path.exists("rocprof_compute_v2_analysis"):
        tools_used.append("ROCprof-compute")

    comprehensive_report["report_metadata"]["tools_used"] = tools_used

    # Save comprehensive report
    with open('v2_comprehensive_analysis.json', 'w') as f:
        json.dump(comprehensive_report, f, indent=2)

    print("Comprehensive analysis complete. Report saved to v2_comprehensive_analysis.json")

    # Print executive summary
    print("\n" + "="*80)
    print("TINY LLAMA V2 - COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)

    fusion_impact = exec_summary.get("fusion_impact", {})
    if "overall_speedup" in fusion_impact:
        print(f"Overall Fusion Speedup: {fusion_impact['overall_speedup']}")

    tool_insights = exec_summary.get("tool_insights", {})
    for tool, insight in tool_insights.items():
        print(f"{tool.replace('_', ' ').title()}: {insight}")

    print("\nKey Findings:")
    for i, finding in enumerate(exec_summary.get("key_findings", [])[:5], 1):
        print(f"{i}. {finding}")

    print("\nOptimization Priorities:")
    for i, priority in enumerate(exec_summary.get("optimization_priorities", [])[:3], 1):
        print(f"{i}. {priority}")

    print(f"\nTools Used: {', '.join(tools_used)}")
    print("="*80)

if __name__ == "__main__":
    main()
EOF

    chmod +x generate_v2_comprehensive_report.py

    # Generate comprehensive report
    if command -v python &> /dev/null; then
        python generate_v2_comprehensive_report.py

        if [ -f "v2_comprehensive_analysis.json" ]; then
            log_info "PASS Comprehensive V2 analysis report generated"
        fi
    else
        log_warning "Python not available - skipping comprehensive report generation"
    fi

    # Generate markdown report
    MARKDOWN_REPORT="v2_comprehensive_report.md"

    cat > "$MARKDOWN_REPORT" << EOF
# Tiny LLaMA V2 - Comprehensive Fusion Analysis Report

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')
**Configuration:** Batch size $BATCH_SIZE, Sequence length $SEQ_LEN, Steps $NUM_STEPS
**Analysis Type:** Complete Fusion Optimization Analysis with ROCm Tools Integration

## Executive Summary

This report presents the most comprehensive analysis of Tiny LLaMA V2 fusion optimizations,
integrating insights from PyTorch Profiler, DeepSpeed FLOPS Profiler, and the complete
ROCm profiling tools suite (ROCprofv3, ROCprof-sys, ROCprof-compute).

## Profiling Tools Integration

### Tools Successfully Used
EOF

    # Add tool status
    if [ "$ENABLE_PYTORCH_PROFILER" = true ] && [ -d "pytorch_v2_profiling" ]; then
        echo "- PASS **PyTorch Profiler** - Framework-level performance analysis" >> "$MARKDOWN_REPORT"
    fi
    if [ "$ENABLE_DEEPSPEED_FLOPS" = true ] && [ -d "deepspeed_v2_flops" ]; then
        echo "- PASS **DeepSpeed FLOPS Profiler** - Computational efficiency analysis" >> "$MARKDOWN_REPORT"
    fi
    if [ "$ENABLE_ROCPROFV3" = true ] && [ -d "rocprofv3_v2_analysis" ]; then
        echo "- PASS **ROCprofv3** - Kernel-level execution analysis" >> "$MARKDOWN_REPORT"
    fi
    if [ "$ENABLE_ROCPROF_SYS" = true ] && [ -d "rocprof_sys_v2_analysis" ]; then
        echo "- PASS **ROCprof-sys** - System-wide resource monitoring" >> "$MARKDOWN_REPORT"
    fi
    if [ "$ENABLE_ROCPROF_COMPUTE" = true ] && [ -d "rocprof_compute_v2_analysis" ]; then
        echo "- PASS **ROCprof-compute** - Advanced optimization recommendations" >> "$MARKDOWN_REPORT"
    fi

    cat >> "$MARKDOWN_REPORT" << EOF

## Performance Results

### Fusion Optimization Impact
EOF

    # Add performance results if available
    if [ -f "v2_comprehensive_analysis.json" ]; then
        # Extract key metrics
        OVERALL_SPEEDUP=$(python -c "import json; data=json.load(open('v2_comprehensive_analysis.json')); print(data.get('executive_summary', {}).get('fusion_impact', {}).get('overall_speedup', 'N/A'))" 2>/dev/null || echo "N/A")
        MFU=$(python -c "import json; data=json.load(open('v2_comprehensive_analysis.json')); print(data.get('executive_summary', {}).get('tool_insights', {}).get('flops_utilization', 'N/A'))" 2>/dev/null || echo "N/A")

        cat >> "$MARKDOWN_REPORT" << EOF

| Metric | Value | Analysis |
|--------|-------|----------|
| **Overall Speedup** | $OVERALL_SPEEDUP | Fusion optimization effectiveness |
| **Model FLOPS Utilization** | $MFU | Computational efficiency |
| **Baseline Speed** | ${BASELINE_SPEED:-N/A} samples/sec | Reference performance |
| **Fused Speed** | ${FUSED_SPEED:-N/A} samples/sec | Optimized performance |
EOF
    fi

    cat >> "$MARKDOWN_REPORT" << EOF

## Analysis Results Directory Structure

\`\`\`
$(pwd)/
├── v2_comprehensive_analysis.json    # Complete analysis results
├── v2_comprehensive_report.md        # This report
├── v2_baseline_performance.log       # Baseline performance metrics
├── v2_fused_performance.log          # Fused performance metrics
EOF

    # List actual directories
    for dir in pytorch_v2_profiling deepspeed_v2_flops rocprofv3_v2_analysis rocprof_sys_v2_analysis rocprof_compute_v2_analysis; do
        if [ -d "$dir" ]; then
            echo "├── $dir/                     # $(echo $dir | sed 's/_v2_/ /' | sed 's/_/ /g' | sed 's/^//' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) substr($i,2)} 1') analysis results" >> "$MARKDOWN_REPORT"
        fi
    done

    cat >> "$MARKDOWN_REPORT" << EOF
└── *.log                            # Detailed execution logs
\`\`\`

## Key Findings and Recommendations

### Fusion Optimization Results
EOF

    # Add key findings from comprehensive analysis
    if [ -f "v2_comprehensive_analysis.json" ]; then
        python -c "
import json
try:
    with open('v2_comprehensive_analysis.json') as f:
        data = json.load(f)
    findings = data.get('executive_summary', {}).get('key_findings', [])
    for i, finding in enumerate(findings[:5], 1):
        print(f'{i}. {finding}')
except:
    print('1. Detailed findings available in comprehensive analysis file')
" >> "$MARKDOWN_REPORT" 2>/dev/null || echo "1. Detailed findings available in comprehensive analysis file" >> "$MARKDOWN_REPORT"
    fi

    cat >> "$MARKDOWN_REPORT" << EOF

### Next Steps for Version 3

Based on this comprehensive analysis:

1. **Implement Custom Triton Kernels** - Address remaining optimization opportunities
2. **Ultra-Fusion Techniques** - Combine multiple operations into single kernels
3. **Hardware-Specific Optimization** - Leverage GPU architecture characteristics
4. **Memory Access Optimization** - Improve bandwidth utilization patterns

## Conclusion

Version 2 demonstrates significant performance improvements through systematic kernel fusion.
The comprehensive profiling analysis provides clear direction for Version 3 optimizations,
targeting custom kernel development and ultra-fusion techniques.

---
*Generated by Castille AI Workshop Comprehensive Profiling Suite*
EOF

    log_info "Comprehensive V2 report generated: $MARKDOWN_REPORT"

else
    log_info "Skipping comprehensive report generation"
fi

# Step 9: Final Analysis Summary
log_step "9. V2 Comprehensive Analysis Complete"

echo ""
echo "Comprehensive V2 fusion analysis completed!"
echo ""
echo "📁 Results Location: $(pwd)"
echo ""
echo "Analysis Summary:"

# Display key results
if [ -f "v2_comprehensive_analysis.json" ]; then
    echo "   Performance Analysis:"
    OVERALL_SPEEDUP=$(python -c "import json; data=json.load(open('v2_comprehensive_analysis.json')); print(data.get('executive_summary', {}).get('fusion_impact', {}).get('overall_speedup', 'N/A'))" 2>/dev/null || echo "N/A")
    echo "      Overall Fusion Speedup: $OVERALL_SPEEDUP"

    if [ "$BASELINE_SPEED" != "N/A" ] && [ "$FUSED_SPEED" != "N/A" ]; then
        echo "      Baseline Performance: $BASELINE_SPEED samples/sec"
        echo "      Fused Performance: $FUSED_SPEED samples/sec"
    fi

    MFU=$(python -c "import json; data=json.load(open('v2_comprehensive_analysis.json')); print(data.get('executive_summary', {}).get('tool_insights', {}).get('flops_utilization', 'N/A'))" 2>/dev/null || echo "N/A")
    if [ "$MFU" != "N/A" ]; then
        echo "      Model FLOPS Utilization: $MFU"
    fi
fi

echo ""
echo "Profiling Tools Used:"
TOOLS_COUNT=0
if [ "$ENABLE_PYTORCH_PROFILER" = true ] && [ -d "pytorch_v2_profiling" ]; then
    echo "   PASS PyTorch Profiler - Framework optimization analysis"
    TOOLS_COUNT=$((TOOLS_COUNT + 1))
fi
if [ "$ENABLE_DEEPSPEED_FLOPS" = true ] && [ -d "deepspeed_v2_flops" ]; then
    echo "   PASS DeepSpeed FLOPS - Computational efficiency analysis"
    TOOLS_COUNT=$((TOOLS_COUNT + 1))
fi
if [ "$ENABLE_ROCPROFV3" = true ] && [ -d "rocprofv3_v2_analysis" ]; then
    echo "   PASS ROCprofv3 - Kernel execution analysis"
    TOOLS_COUNT=$((TOOLS_COUNT + 1))
fi
if [ "$ENABLE_ROCPROF_SYS" = true ] && [ -d "rocprof_sys_v2_analysis" ]; then
    echo "   PASS ROCprof-sys - System resource monitoring"
    TOOLS_COUNT=$((TOOLS_COUNT + 1))
fi
if [ "$ENABLE_ROCPROF_COMPUTE" = true ] && [ -d "rocprof_compute_v2_analysis" ]; then
    echo "   PASS ROCprof-compute - Advanced optimization recommendations"
    TOOLS_COUNT=$((TOOLS_COUNT + 1))
fi

echo "   Total profiling tools used: $TOOLS_COUNT"

echo ""
echo "Generated Reports:"
if [ -f "v2_comprehensive_analysis.json" ]; then
    echo "   - v2_comprehensive_analysis.json    # Complete analysis data"
fi
if [ -f "v2_comprehensive_report.md" ]; then
    echo "   - v2_comprehensive_report.md        # Executive summary report"
fi
echo "   - Individual tool reports in respective directories"

echo ""
echo "🔄 Next Steps:"
echo "   1. Review comprehensive analysis: cat v2_comprehensive_report.md"
echo "   2. Examine individual tool results for detailed insights"
echo "   3. Compare V2 results with V1 baseline measurements"
echo "   4. Plan Version 3 custom kernel optimizations"
echo "   5. Use ROCprof-compute recommendations for Triton development"

echo ""
echo "Advanced Analysis Commands:"
echo "   # Compare with V1 (if available)"
echo "   python compare_v1_v2_results.py --v1 ../version1_pytorch_baseline/complete_analysis --v2 ."
echo ""
echo "   # Launch TensorBoard for PyTorch profiling visualization"
echo "   tensorboard --logdir pytorch_v2_profiling --port 6006"
echo ""
echo "   # Detailed ROCprof-compute analysis"
echo "   rocprof-compute analyze rocprof_compute_v2_analysis/fused_advanced/"
echo ""

log_info "V2 comprehensive profiling analysis complete!"
log_analysis "Ready to proceed to Version 3 with optimization guidance from $TOOLS_COUNT profiling tools"

# Return to original directory
cd - > /dev/null