#!/bin/bash

# This test runs an example from https://github.com/CliMA/Oceananigans.jl

# NOTE: this test assumes Julia has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/julia_setup.sh

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

ORIG_HOME="${HOME}"

export WORK_DIR=$(mktemp -d)
trap "rm -rf ${WORK_DIR}" EXIT
cd ${WORK_DIR}
export HOME="${WORK_DIR}"
export JULIA_DEPOT_PATH="${WORK_DIR}/julia_depot"
export JULIA_NUM_PRECOMPILE_TASKS=8
curl -fsSL https://install.julialang.org | sh -s -- --yes --add-to-path=no -p=${WORK_DIR}/juliaup_install
export PATH=$PATH:"${WORK_DIR}/juliaup_install/bin"

juliaup add 1.12
juliaup default 1.12

git clone https://github.com/CliMA/Oceananigans.jl.git &
CLONE_PID=$!

julia -e 'using Pkg; Pkg.add(["AMDGPU", "MPI", "Oceananigans", "CUDA", "FFTW", "KernelAbstractions", "SeawaterPolynomials", "OffsetArrays", "JLD2", "Adapt", "GPUArraysCore"])'

wait $CLONE_PID
pushd Oceananigans.jl/test
julia test_amdgpu.jl
popd
rm -rf Oceananigans.jl

export HOME="${ORIG_HOME}"
