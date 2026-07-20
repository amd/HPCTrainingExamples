#!/usr/bin/env bash
# Source this to activate the ROCm venv and set ROCm env vars:
#   source setup_rocm.sh
VENV="${HOME}/venvs/rocm-pytorch-pip"
DEVEL="${VENV}/lib/python3.12/site-packages/_rocm_sdk_devel"
source "${VENV}/bin/activate"
export ROCM_PATH="$DEVEL"
export HIP_PATH="$DEVEL"
export HIP_DEVICE_LIB_PATH="${DEVEL}/lib/llvm/amdgcn/bitcode"
export PATH="${DEVEL}/bin:$PATH"
export LD_LIBRARY_PATH="${DEVEL}/lib:${DEVEL}/lib/rocm_sysdeps/lib:$LD_LIBRARY_PATH"
echo "ROCm venv active: $VENV"

