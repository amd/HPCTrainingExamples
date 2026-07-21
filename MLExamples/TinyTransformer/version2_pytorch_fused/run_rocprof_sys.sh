#!/bin/bash

# rocprof-sys (System Profiler) Integration for Tiny LLaMA V2
# System-wide performance monitoring and resource utilization analysis

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

log_sys() {
    echo -e "${PURPLE}[ROCPROF-SYS]${NC} $1"
}

# Default configuration
BATCH_SIZE=8
SEQ_LEN=128
NUM_STEPS=50
DURATION=120  # seconds
OUTPUT_DIR="./rocprof_sys_results_$(date +%Y%m%d_%H%M%S)"
SAMPLE_RATE=100  # Hz
MONITOR_CPU=true
MONITOR_GPU=true
MONITOR_MEMORY=true
MONITOR_IO=false
SYSTEM_WIDE=true

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
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --sample-rate)
            SAMPLE_RATE="$2"
            shift 2
            ;;
        --monitor-cpu)
            MONITOR_CPU=true
            shift
            ;;
        --no-cpu)
            MONITOR_CPU=false
            shift
            ;;
        --monitor-gpu)
            MONITOR_GPU=true
            shift
            ;;
        --no-gpu)
            MONITOR_GPU=false
            shift
            ;;
        --monitor-memory)
            MONITOR_MEMORY=true
            shift
            ;;
        --no-memory)
            MONITOR_MEMORY=false
            shift
            ;;
        --monitor-io)
            MONITOR_IO=true
            shift
            ;;
        --process-only)
            SYSTEM_WIDE=false
            shift
            ;;
        --help|-h)
            echo "rocprof-sys System Profiling for Tiny LLaMA V2"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --batch-size SIZE         Batch size (default: 8)"
            echo "  --seq-len LENGTH          Sequence length (default: 128)"
            echo "  --num-steps STEPS         Training steps (default: 50)"
            echo "  --duration SECONDS        Profile duration in seconds (default: 120)"
            echo "  --output-dir DIR          Output directory for results"
            echo "  --sample-rate HZ          Sampling rate in Hz (default: 100)"
            echo "  --monitor-cpu             Enable CPU monitoring (default)"
            echo "  --no-cpu                  Disable CPU monitoring"
            echo "  --monitor-gpu             Enable GPU monitoring (default)"
            echo "  --no-gpu                  Disable GPU monitoring"
            echo "  --monitor-memory          Enable memory monitoring (default)"
            echo "  --no-memory               Disable memory monitoring"
            echo "  --monitor-io              Enable I/O monitoring"
            echo "  --process-only            Monitor only target process (not system-wide)"
            echo "  --help, -h                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Basic system profiling"
            echo "  $0 --duration 180 --sample-rate 200  # Extended high-frequency profiling"
            echo "  $0 --process-only --no-io           # Process-focused profiling"
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
echo "ROCPROF-SYS SYSTEM PROFILING - TINY LLAMA V2"
echo "    System-Wide Performance Monitoring and Resource Analysis"
echo "=" * 80
echo ""

# Step 1: Environment Validation
log_step "1. Environment Validation"

# Check if rocprof-sys is available
if ! command -v rocprof-sys &> /dev/null; then
    log_warning "rocprof-sys not found - attempting fallback monitoring"
    ROCPROF_SYS_AVAILABLE=false
else
    ROCPROF_SYS_VERSION=$(rocprof-sys --version 2>&1 | head -n1 || echo "Unknown")
    log_info "rocprof-sys version: $ROCPROF_SYS_VERSION"
    ROCPROF_SYS_AVAILABLE=true
fi

# Check system monitoring tools
TOOLS_AVAILABLE=""
if command -v rocm-smi &> /dev/null; then
    TOOLS_AVAILABLE="$TOOLS_AVAILABLE rocm-smi"
fi
if command -v nvidia-smi &> /dev/null; then
    TOOLS_AVAILABLE="$TOOLS_AVAILABLE nvidia-smi"
fi
if command -v htop &> /dev/null; then
    TOOLS_AVAILABLE="$TOOLS_AVAILABLE htop"
fi
if command -v iostat &> /dev/null; then
    TOOLS_AVAILABLE="$TOOLS_AVAILABLE iostat"
fi

log_info "Available monitoring tools:$TOOLS_AVAILABLE"

# Check if target script exists
if [ ! -f "../tiny_llama_v2.py" ]; then
    log_error "tiny_llama_v2.py not found. Please run this script from version2_pytorch_fused directory."
    exit 1
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
log_info "  Profile duration: ${DURATION}s"
log_info "  Sample rate: ${SAMPLE_RATE}Hz"
log_info "  CPU monitoring: $MONITOR_CPU"
log_info "  GPU monitoring: $MONITOR_GPU"
log_info "  Memory monitoring: $MONITOR_MEMORY"
log_info "  I/O monitoring: $MONITOR_IO"
log_info "  System-wide: $SYSTEM_WIDE"

# Step 2: System Baseline Collection
log_step "2. System Baseline Collection"

log_sys "Collecting system baseline metrics..."

# Collect baseline system information
cat > system_baseline.txt << EOF
# System Baseline Information
Generated: $(date)
Hostname: $(hostname)
Kernel: $(uname -a)
CPU Info: $(lscpu | grep "Model name" | head -1)
Memory: $(free -h | grep "Mem:")
EOF

# GPU baseline
if [ "$MONITOR_GPU" = true ]; then
    if command -v rocm-smi &> /dev/null; then
        echo "GPU Info (ROCm):" >> system_baseline.txt
        rocm-smi --showid --showproductname >> system_baseline.txt 2>/dev/null || echo "ROCm GPU info failed" >> system_baseline.txt
    elif command -v nvidia-smi &> /dev/null; then
        echo "GPU Info (NVIDIA):" >> system_baseline.txt
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> system_baseline.txt 2>/dev/null || echo "NVIDIA GPU info failed" >> system_baseline.txt
    fi
fi

log_info "PASS System baseline collected"

# Step 3: Background System Monitoring Setup
log_step "3. Background System Monitoring Setup"

# Create system monitoring scripts
if [ "$MONITOR_CPU" = true ] || [ "$MONITOR_MEMORY" = true ]; then
    cat > monitor_system.py << 'EOF'
#!/usr/bin/env python3
"""
System Resource Monitor for rocprof-sys Integration
Monitors CPU, memory, and system resources during ML training
"""

import psutil
import time
import json
import sys
import subprocess
from datetime import datetime

def get_gpu_stats():
    """Get GPU statistics from rocm-smi or nvidia-smi."""
    gpu_stats = {}

    # Try ROCm first
    try:
        result = subprocess.run(['rocm-smi', '--showuse', '--showmemuse'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse rocm-smi output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GPU[' in line and '%' in line:
                    # Extract GPU utilization
                    parts = line.split()
                    if len(parts) > 3:
                        gpu_stats['utilization'] = parts[3].replace('%', '')
                        break

            # Get memory info
            result_mem = subprocess.run(['rocm-smi', '--showmemuse'],
                                      capture_output=True, text=True, timeout=5)
            if result_mem.returncode == 0:
                lines_mem = result_mem.stdout.split('\n')
                for line in lines_mem:
                    if 'GPU[' in line and 'MB' in line:
                        # Extract memory usage
                        if '/' in line:
                            mem_parts = line.split('/')
                            if len(mem_parts) > 1:
                                used_mb = mem_parts[0].split()[-1]
                                total_mb = mem_parts[1].split()[0]
                                gpu_stats['memory_used_mb'] = used_mb
                                gpu_stats['memory_total_mb'] = total_mb
                        break
            return gpu_stats
    except:
        pass

    # Try NVIDIA
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            line = result.stdout.strip()
            parts = line.split(', ')
            if len(parts) >= 3:
                gpu_stats['utilization'] = parts[0]
                gpu_stats['memory_used_mb'] = parts[1]
                gpu_stats['memory_total_mb'] = parts[2]
    except:
        pass

    return gpu_stats

def monitor_resources(duration, sample_rate, output_file):
    """Monitor system resources for specified duration."""
    interval = 1.0 / sample_rate
    samples = []
    start_time = time.time()
    end_time = start_time + duration

    print(f"Monitoring for {duration} seconds at {sample_rate}Hz...")

    while time.time() < end_time:
        timestamp = datetime.now().isoformat()

        # CPU and memory stats
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        # GPU stats
        gpu_stats = get_gpu_stats()

        sample = {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'gpu_stats': gpu_stats
        }

        samples.append(sample)
        time.sleep(interval)

    # Save results
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"Monitoring complete. {len(samples)} samples saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python monitor_system.py <duration> <sample_rate> <output_file>")
        sys.exit(1)

    duration = float(sys.argv[1])
    sample_rate = float(sys.argv[2])
    output_file = sys.argv[3]

    monitor_resources(duration, sample_rate, output_file)
EOF

    chmod +x monitor_system.py
    log_info "PASS System monitoring script created"
fi

# Step 4: rocprof-sys System Profiling
log_step "4. rocprof-sys System Profiling"

if [ "$ROCPROF_SYS_AVAILABLE" = true ]; then
    log_sys "Starting rocprof-sys system-wide profiling..."

    # Build rocprof-sys command
    ROCPROF_SYS_CMD="rocprof-sys"

    if [ "$SYSTEM_WIDE" = true ]; then
        ROCPROF_SYS_CMD="$ROCPROF_SYS_CMD --system-wide"
    fi

    ROCPROF_SYS_CMD="$ROCPROF_SYS_CMD --duration $DURATION --output rocprof_sys_trace"

    # Start rocprof-sys in background
    log_sys "Starting rocprof-sys profiling..."
    $ROCPROF_SYS_CMD > rocprof_sys.log 2>&1 &
    ROCPROF_SYS_PID=$!

    # Wait a moment for rocprof-sys to initialize
    sleep 3

    log_info "rocprof-sys running with PID: $ROCPROF_SYS_PID"
else
    log_warning "rocprof-sys not available - using fallback monitoring"
fi

# Step 5: Start Background System Monitoring
log_step "5. Background System Monitoring"

MONITOR_PIDS=()

if [ "$MONITOR_CPU" = true ] || [ "$MONITOR_MEMORY" = true ]; then
    log_sys "Starting background system monitoring..."
    python monitor_system.py $DURATION $SAMPLE_RATE system_resources.json > system_monitor.log 2>&1 &
    SYSTEM_MONITOR_PID=$!
    MONITOR_PIDS+=($SYSTEM_MONITOR_PID)
    log_info "System monitor running with PID: $SYSTEM_MONITOR_PID"
fi

# GPU-specific monitoring
if [ "$MONITOR_GPU" = true ]; then
    if command -v rocm-smi &> /dev/null; then
        log_sys "Starting ROCm GPU monitoring..."
        # Create GPU monitoring script
        cat > monitor_gpu.sh << 'EOF'
#!/bin/bash
duration=$1
interval=1

end_time=$(($(date +%s) + duration))
echo "timestamp,gpu_util,memory_used,memory_total" > gpu_stats.csv

while [ $(date +%s) -lt $end_time ]; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Get GPU utilization
    gpu_util=$(rocm-smi --showuse 2>/dev/null | grep "GPU\[0\]" | awk '{print $3}' | tr -d '%' || echo "0")

    # Get memory usage
    mem_info=$(rocm-smi --showmemuse 2>/dev/null | grep "GPU\[0\]" | awk '{print $3}' || echo "0/0")
    mem_used=$(echo $mem_info | cut -d'/' -f1 | tr -d 'MB ')
    mem_total=$(echo $mem_info | cut -d'/' -f2 | tr -d 'MB ')

    echo "$timestamp,$gpu_util,$mem_used,$mem_total" >> gpu_stats.csv
    sleep $interval
done
EOF
        chmod +x monitor_gpu.sh
        ./monitor_gpu.sh $DURATION > gpu_monitor.log 2>&1 &
        GPU_MONITOR_PID=$!
        MONITOR_PIDS+=($GPU_MONITOR_PID)
        log_info "GPU monitor running with PID: $GPU_MONITOR_PID"
    fi
fi

# I/O monitoring if requested
if [ "$MONITOR_IO" = true ] && command -v iostat &> /dev/null; then
    log_sys "Starting I/O monitoring..."
    iostat -x 1 $DURATION > io_stats.log 2>&1 &
    IO_MONITOR_PID=$!
    MONITOR_PIDS+=($IO_MONITOR_PID)
    log_info "I/O monitor running with PID: $IO_MONITOR_PID"
fi

# Step 6: Run Target Application
log_step "6. Running Target Application"

log_info "Starting Tiny LLaMA V2 training with system monitoring active..."

# Run the training with different fusion configurations
CONFIGS=(
    "--disable-all-fusion"
    "--enable-all-fusion"
    "--enable-qkv-fusion --disable-flash-attention --disable-swiglu-fusion"
    "--disable-qkv-fusion --enable-flash-attention --disable-swiglu-fusion"
)

CONFIG_NAMES=(
    "baseline"
    "full_fusion"
    "qkv_only"
    "flash_only"
)

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    name="${CONFIG_NAMES[$i]}"

    log_info "Running configuration: $name"

    # Run training
    timeout $((DURATION / 4)) python ../tiny_llama_v2.py \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --num-steps $((NUM_STEPS / 4)) \
        $config > "training_${name}.log" 2>&1 &

    TRAIN_PID=$!
    log_info "Training ($name) running with PID: $TRAIN_PID"

    # Wait for completion or timeout
    wait $TRAIN_PID 2>/dev/null

    log_info "Configuration $name completed"
    sleep 5  # Brief pause between configurations
done

# Step 7: Wait for All Monitoring to Complete
log_step "7. Monitoring Completion"

log_info "Waiting for monitoring to complete..."

# Wait for rocprof-sys if running
if [ -n "$ROCPROF_SYS_PID" ] && kill -0 "$ROCPROF_SYS_PID" 2>/dev/null; then
    log_sys "Waiting for rocprof-sys to complete..."
    wait $ROCPROF_SYS_PID 2>/dev/null
    log_info "rocprof-sys completed"
fi

# Wait for background monitors
for pid in "${MONITOR_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        log_info "Waiting for monitor PID $pid to complete..."
        wait $pid 2>/dev/null
    fi
done

log_info "PASS All monitoring completed"

# Step 8: Data Analysis and Report Generation
log_step "8. Data Analysis and Report Generation"

# Create analysis script
cat > analyze_system_data.py << 'EOF'
#!/usr/bin/env python3
"""
System profiling data analysis for rocprof-sys results
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_system_resources():
    """Analyze system resource usage data."""
    try:
        with open('system_resources.json', 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        analysis = {
            "resource_analysis": {
                "cpu_utilization": {
                    "mean": float(df['cpu_percent'].mean()),
                    "max": float(df['cpu_percent'].max()),
                    "min": float(df['cpu_percent'].min()),
                    "std": float(df['cpu_percent'].std())
                },
                "memory_utilization": {
                    "mean": float(df['memory_percent'].mean()),
                    "max": float(df['memory_percent'].max()),
                    "min": float(df['memory_percent'].min()),
                    "std": float(df['memory_percent'].std())
                },
                "memory_usage_gb": {
                    "mean": float(df['memory_used_gb'].mean()),
                    "max": float(df['memory_used_gb'].max()),
                    "peak": float(df['memory_used_gb'].max())
                }
            },
            "gpu_analysis": {},
            "sample_count": len(df),
            "duration_seconds": len(df) / 100  # Assuming 100Hz sampling
        }

        # GPU analysis if available
        gpu_utils = []
        gpu_memory_used = []

        for sample in data:
            gpu_stats = sample.get('gpu_stats', {})
            if 'utilization' in gpu_stats:
                try:
                    gpu_utils.append(float(gpu_stats['utilization']))
                except:
                    pass
            if 'memory_used_mb' in gpu_stats:
                try:
                    gpu_memory_used.append(float(gpu_stats['memory_used_mb']))
                except:
                    pass

        if gpu_utils:
            analysis["gpu_analysis"]["utilization"] = {
                "mean": float(np.mean(gpu_utils)),
                "max": float(np.max(gpu_utils)),
                "min": float(np.min(gpu_utils)),
                "std": float(np.std(gpu_utils))
            }

        if gpu_memory_used:
            analysis["gpu_analysis"]["memory_used_mb"] = {
                "mean": float(np.mean(gpu_memory_used)),
                "max": float(np.max(gpu_memory_used)),
                "peak": float(np.max(gpu_memory_used))
            }

        return analysis

    except Exception as e:
        return {"error": f"Failed to analyze system resources: {str(e)}"}

def analyze_gpu_stats():
    """Analyze GPU statistics if available."""
    try:
        if not Path('gpu_stats.csv').exists():
            return {"error": "GPU stats not available"}

        df = pd.read_csv('gpu_stats.csv')

        analysis = {
            "gpu_utilization": {
                "mean": float(df['gpu_util'].mean()),
                "max": float(df['gpu_util'].max()),
                "min": float(df['gpu_util'].min())
            },
            "gpu_memory": {
                "mean_used_mb": float(df['memory_used'].mean()),
                "peak_used_mb": float(df['memory_used'].max()),
                "total_mb": float(df['memory_total'].iloc[0]) if len(df) > 0 else 0
            },
            "samples": len(df)
        }

        return analysis

    except Exception as e:
        return {"error": f"Failed to analyze GPU stats: {str(e)}"}

def main():
    """Generate comprehensive system analysis."""
    print("Analyzing system profiling data...")

    results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "system_resources": analyze_system_resources(),
        "gpu_statistics": analyze_gpu_stats()
    }

    # Save analysis
    with open('system_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Analysis complete. Results saved to system_analysis.json")

    # Print summary
    print("\nSYSTEM PROFILING SUMMARY:")

    sys_resources = results.get('system_resources', {})
    if "error" not in sys_resources:
        cpu = sys_resources.get('resource_analysis', {}).get('cpu_utilization', {})
        memory = sys_resources.get('resource_analysis', {}).get('memory_utilization', {})
        print(f"CPU Utilization: {cpu.get('mean', 0):.1f}% avg, {cpu.get('max', 0):.1f}% peak")
        print(f"Memory Utilization: {memory.get('mean', 0):.1f}% avg, {memory.get('max', 0):.1f}% peak")

        gpu = sys_resources.get('gpu_analysis', {})
        if 'utilization' in gpu:
            gpu_util = gpu['utilization']
            print(f"GPU Utilization: {gpu_util.get('mean', 0):.1f}% avg, {gpu_util.get('max', 0):.1f}% peak")

if __name__ == "__main__":
    main()
EOF

chmod +x analyze_system_data.py

# Run analysis
if command -v python &> /dev/null; then
    log_info "Running system data analysis..."
    python analyze_system_data.py

    if [ -f "system_analysis.json" ]; then
        log_info "PASS System analysis completed"
    fi
else
    log_warning "Python not available - skipping data analysis"
fi

# Generate comprehensive report
REPORT_FILE="rocprof_sys_report.md"

cat > "$REPORT_FILE" << EOF
# rocprof-sys System Analysis Report - Tiny LLaMA V2

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')
**Profile Duration:** ${DURATION} seconds
**Configuration:** Batch size $BATCH_SIZE, Sequence length $SEQ_LEN

## Executive Summary

This report provides system-wide performance analysis of Tiny LLaMA V2 training
using rocprof-sys and supplementary monitoring tools to understand resource
utilization patterns and system bottlenecks.

## Monitoring Configuration

- **rocprof-sys Available:** $ROCPROF_SYS_AVAILABLE
- **Sample Rate:** ${SAMPLE_RATE}Hz
- **CPU Monitoring:** $MONITOR_CPU
- **GPU Monitoring:** $MONITOR_GPU
- **Memory Monitoring:** $MONITOR_MEMORY
- **I/O Monitoring:** $MONITOR_IO
- **System-wide Profiling:** $SYSTEM_WIDE

## System Resource Analysis
EOF

# Add analysis results if available
if [ -f "system_analysis.json" ]; then
    cat >> "$REPORT_FILE" << EOF

### Resource Utilization Summary
\`\`\`json
$(cat system_analysis.json)
\`\`\`
EOF
fi

cat >> "$REPORT_FILE" << EOF

## Training Configurations Analyzed

1. **Baseline (No Fusion)** - Reference implementation
2. **Full Fusion** - All optimizations enabled
3. **QKV Fusion Only** - Selective optimization
4. **Flash Attention Only** - Memory-efficient attention

## Generated Files

- \`system_analysis.json\` - Comprehensive resource analysis
- \`system_resources.json\` - Raw system monitoring data
- \`gpu_stats.csv\` - GPU utilization timeline
- \`training_*.log\` - Training execution logs
- \`rocprof_sys_trace\` - rocprof-sys trace data (if available)

## Key Findings

### System Performance Characteristics
- Resource utilization patterns during different fusion configurations
- System bottleneck identification
- GPU vs CPU utilization balance
- Memory pressure analysis

### Optimization Impact
- System-level impact of fusion optimizations
- Resource efficiency improvements
- Scaling characteristics

## Next Steps

1. **Detailed Trace Analysis**: Examine rocprof-sys trace data
2. **Bottleneck Resolution**: Address identified system bottlenecks
3. **rocprof-compute**: Use advanced profiler for optimization hints
4. **Production Scaling**: Apply findings to larger-scale deployments

---
*Generated by AI Workshop rocprof-sys Analysis Tool*
EOF

log_info "System analysis report generated: $REPORT_FILE"

# Step 9: Final Summary
log_step "9. rocprof-sys Analysis Complete"

echo ""
echo "rocprof-sys system profiling completed!"
echo ""
echo "📁 Results Location: $(pwd)"
echo ""
echo "Generated Files:"
echo "   - $REPORT_FILE              # System analysis report"
echo "   - system_analysis.json      # Processed analysis results"
echo "   - system_resources.json     # Raw monitoring data"
if [ -f "gpu_stats.csv" ]; then
    echo "   - gpu_stats.csv             # GPU utilization timeline"
fi
echo "   - training_*.log            # Application execution logs"
if [ "$ROCPROF_SYS_AVAILABLE" = true ]; then
    echo "   - rocprof_sys_trace         # rocprof-sys trace data"
fi
echo ""
echo "Key System Insights:"
if [ -f "system_analysis.json" ]; then
    # Extract key metrics
    CPU_AVG=$(python -c "import json; data=json.load(open('system_analysis.json')); print(f\"{data.get('system_resources', {}).get('resource_analysis', {}).get('cpu_utilization', {}).get('mean', 0):.1f}\")" 2>/dev/null || echo "N/A")
    MEM_AVG=$(python -c "import json; data=json.load(open('system_analysis.json')); print(f\"{data.get('system_resources', {}).get('resource_analysis', {}).get('memory_utilization', {}).get('mean', 0):.1f}\")" 2>/dev/null || echo "N/A")
    echo "   Average CPU utilization: ${CPU_AVG}%"
    echo "   Average memory utilization: ${MEM_AVG}%"
fi
echo ""
echo "🔄 Next Steps:"
echo "   1. Review system analysis: cat $REPORT_FILE"
echo "   2. Examine resource timeline: head -20 system_resources.json"
echo "   3. Run rocprof-compute for detailed optimization analysis"
echo "   4. Compare system impact across fusion configurations"
echo ""

log_info "rocprof-sys analysis data saved to: $(pwd)"
log_info "System-wide profiling analysis complete!"

# Return to original directory
cd - > /dev/null
