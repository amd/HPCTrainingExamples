#!/bin/bash
# Run TinyLlama V1 baseline without profiling

set -e

echo "========================================================================"
echo "Running TinyLlama V1 - Baseline (No Profiling)"
echo "========================================================================"

python tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 50

echo ""
echo "Baseline run completed!"
