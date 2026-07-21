#!/usr/bin/env bash
# Source this to activate the ROCm venv and set ROCm env vars:
#   source setup_rocm.sh
VENV="${HOME}/venvs/rocm-pytorch-pip"
source "${VENV}/bin/activate"
# Derive site-packages from the active venv so this works regardless of the
# venv's Python minor version (e.g. python3.12 vs python3.13).
DEVEL="$(python3 -c 'import site; print(site.getsitepackages()[0])')/_rocm_sdk_devel"
export ROCM_PATH="$DEVEL"
export HIP_PATH="$DEVEL"
export HIP_DEVICE_LIB_PATH="${DEVEL}/lib/llvm/amdgcn/bitcode"
export PATH="${DEVEL}/bin:$PATH"
export LD_LIBRARY_PATH="${DEVEL}/lib:${DEVEL}/lib/rocm_sysdeps/lib:$LD_LIBRARY_PATH"
echo "ROCm venv active: $VENV"

