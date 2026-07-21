#!/bin/bash
# Run TinyLlama V1 with PyTorch Profiler

set -e

echo "========================================================================"
echo "Running TinyLlama V1 - PyTorch Profiler"
echo "========================================================================"

# Create profile directory
mkdir -p pytorch_profiles

python run_pytorch_profiler.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 20 \
    --profile-steps 5 \
    --profile-dir pytorch_profiles \
    --include-memory

echo ""
echo "PyTorch profiler run completed!"
echo "Profile data saved to: pytorch_profiles/"
echo "Launch TensorBoard: tensorboard --logdir pytorch_profiles"
