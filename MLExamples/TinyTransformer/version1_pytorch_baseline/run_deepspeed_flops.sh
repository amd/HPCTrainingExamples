#!/bin/bash
# Run TinyLlama V1 with DeepSpeed FLOPS Profiler

set -e

echo "========================================================================"
echo "Running TinyLlama V1 - DeepSpeed FLOPS Profiler"
echo "========================================================================"

# Create profile directory
mkdir -p deepspeed_profiles

python run_deepspeed_flops.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 20 \
    --profile-dir deepspeed_profiles

echo ""
echo "DeepSpeed FLOPS profiler run completed!"
echo "Profile data saved to: deepspeed_profiles/"
