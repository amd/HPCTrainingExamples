#!/bin/bash

# rocprof-compute Profiling Integration for Tiny OpenFold V2
# This script provides detailed hardware-level profiling and roofline analysis

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
log_rocprof() { echo -e "${PURPLE}[ROCPROF-COMPUTE]${NC} $1"; }

# Default configuration
BATCH_SIZE=4
SEQ_LEN=64
NUM_BLOCKS=4
NUM_SEQS=16
NUM_STEPS=30
OUTPUT_NAME="tinyfold_v2"
MODE="profile"  # profile, roof, or analyze
DEVICE=0
ROOF_ONLY=false
NO_ROOF=false
DISPATCH_ID=""
ENABLE_ALL_FUSION=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --seq-len) SEQ_LEN="$2"; shift 2 ;;
        --num-blocks) NUM_BLOCKS="$2"; shift 2 ;;
        --num-seqs) NUM_SEQS="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        --output-name) OUTPUT_NAME="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --roof-only) ROOF_ONLY=true; shift ;;
        --no-roof) NO_ROOF=true; shift ;;
        --dispatch) DISPATCH_ID="$2"; shift 2 ;;
        --disable-all-fusion) ENABLE_ALL_FUSION=false; shift ;;
        --help|-h)
            echo "rocprof-compute Profiling for Tiny OpenFold V2"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Modes:"
            echo "  --mode profile          Profile and collect data (default)"
            echo "  --mode roof             Generate roofline plots only"
            echo "  --mode analyze          Analyze specific dispatch"
            echo ""
            echo "Options:"
            echo "  --batch-size N          Batch size (default: 4)"
            echo "  --seq-len N             Sequence length (default: 64)"
            echo "  --num-blocks N          Number of Evoformer blocks (default: 4)"
            echo "  --num-seqs N            Number of MSA sequences (default: 16)"
            echo "  --num-steps N           Training steps (default: 30)"
            echo "  --output-name NAME      Output name (default: tinyfold_v2)"
            echo "  --device N              GPU device (default: 0)"
            echo "  --roof-only             Generate roofline only (faster)"
            echo "  --no-roof               Skip roofline generation"
            echo "  --dispatch ID           Analyze specific dispatch ID"
            echo "  --disable-all-fusion    Disable all fusions"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Full profile with roofline"
            echo "  $0 --roof-only                        # Roofline only (faster)"
            echo "  $0 --no-roof                          # Profile without roofline"
            echo "  $0 --mode analyze --dispatch 1538     # Analyze specific dispatch"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check for rocprof-compute
if ! command -v rocprof-compute &> /dev/null; then
    log_info "rocprof-compute not found. Please ensure ROCm tools are installed."
    exit 1
fi

log_info "======================================================================"
log_info "Tiny OpenFold V2 - rocprof-compute Profiling"
log_info "======================================================================"
echo ""
log_info "Configuration:"
log_info "  Mode: $MODE"
log_info "  Batch size: $BATCH_SIZE"
log_info "  Sequence length: $SEQ_LEN"
log_info "  Evoformer blocks: $NUM_BLOCKS"
log_info "  MSA sequences: $NUM_SEQS"
log_info "  Training steps: $NUM_STEPS"
log_info "  All fusions: $ENABLE_ALL_FUSION"
log_info "  Device: $DEVICE"
log_info "  Output name: $OUTPUT_NAME"
echo ""

# Build Python command
PYTHON_ARGS="--batch-size $BATCH_SIZE --seq-len $SEQ_LEN --num-blocks $NUM_BLOCKS --num-seqs $NUM_SEQS --num-steps $NUM_STEPS"
[ "$ENABLE_ALL_FUSION" = false ] && PYTHON_ARGS="$PYTHON_ARGS --disable-all-fusion"

case $MODE in
    profile)
        log_step "Running rocprof-compute profile..."
        
        if [ "$ROOF_ONLY" = true ]; then
            log_rocprof "Mode: Roofline only (faster profiling)"
            rocprof-compute profile -n $OUTPUT_NAME --kernel-names --roof-only --device $DEVICE \
                -- python tiny_openfold_v2.py $PYTHON_ARGS 2>&1 | tee rocprof_compute_roof.log
        elif [ "$NO_ROOF" = true ]; then
            log_rocprof "Mode: Full profile without roofline"
            rocprof-compute profile -n $OUTPUT_NAME --no-roof --device $DEVICE \
                -- python tiny_openfold_v2.py $PYTHON_ARGS 2>&1 | tee rocprof_compute_profile.log
        else
            log_rocprof "Mode: Full profile with roofline"
            rocprof-compute profile -n $OUTPUT_NAME --device $DEVICE \
                -- python tiny_openfold_v2.py $PYTHON_ARGS 2>&1 | tee rocprof_compute_full.log
        fi
        
        log_step "Profiling complete!"
        
        # Check for generated files
        echo ""
        log_info "Generated files:"
        
        # Roofline PDFs
        if [ "$NO_ROOF" = false ]; then
            if ls roofline_*.pdf 1> /dev/null 2>&1; then
                log_info "  Roofline plots:"
                ls -lh roofline_*.pdf | awk '{print "    - " $9 " (" $5 ")"}'
            fi
        fi
        
        # Workload directory
        if [ -d "workloads/${OUTPUT_NAME}" ]; then
            log_info "  Workload data: workloads/${OUTPUT_NAME}/"
        fi
        
        # Suggest next steps
        echo ""
        log_info "Next steps:"
        log_info "  1. View roofline plots: open roofline_*.pdf"
        log_info "  2. List dispatches: rocprof-compute analyze -p workloads/${OUTPUT_NAME}/* --list-stats"
        log_info "  3. Analyze dispatch: $0 --mode analyze --dispatch <ID>"
        ;;
        
    roof)
        log_step "Generating roofline plots..."
        rocprof-compute profile -n $OUTPUT_NAME --kernel-names --roof-only --device $DEVICE \
            -- python tiny_openfold_v2.py $PYTHON_ARGS 2>&1 | tee rocprof_compute_roof.log
        
        log_step "Roofline generation complete!"
        
        if ls roofline_*.pdf 1> /dev/null 2>&1; then
            echo ""
            log_info "Generated roofline plots:"
            ls -lh roofline_*.pdf
        fi
        ;;
        
    analyze)
        if [ -z "$DISPATCH_ID" ]; then
            log_info "Listing available dispatches..."
            WORKLOAD_DIR=$(find workloads/${OUTPUT_NAME} -type d -name "MI*" | head -n 1)
            
            if [ -z "$WORKLOAD_DIR" ]; then
                log_info "No workload data found. Run with --mode profile first."
                exit 1
            fi
            
            rocprof-compute analyze -p $WORKLOAD_DIR --list-stats > dispatch_list.txt 2>&1
            
            echo ""
            log_info "Available dispatches saved to: dispatch_list.txt"
            echo ""
            head -n 50 dispatch_list.txt
            echo ""
            log_info "To analyze a specific dispatch:"
            log_info "  $0 --mode analyze --dispatch <ID>"
        else
            log_step "Analyzing dispatch $DISPATCH_ID..."
            WORKLOAD_DIR=$(find workloads/${OUTPUT_NAME} -type d -name "MI*" | head -n 1)
            
            if [ -z "$WORKLOAD_DIR" ]; then
                log_info "No workload data found. Run with --mode profile first."
                exit 1
            fi
            
            rocprof-compute analyze -p $WORKLOAD_DIR --dispatch $DISPATCH_ID > dispatch_${DISPATCH_ID}_analysis.txt 2>&1
            
            log_step "Analysis complete!"
            echo ""
            log_info "Analysis saved to: dispatch_${DISPATCH_ID}_analysis.txt"
            echo ""
            head -n 100 dispatch_${DISPATCH_ID}_analysis.txt
        fi
        ;;
        
    *)
        log_info "Unknown mode: $MODE"
        log_info "Use --help for usage information"
        exit 1
        ;;
esac

echo ""
log_info "======================================================================"
log_info "rocprof-compute Complete!"
log_info "======================================================================"
echo ""


