#!/bin/bash

# This test runs an example from https://github.com/CliMA/Oceananigans.jl

# NOTE: this test assumes Julia has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/julia_setup.sh

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
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

julia -e '
using Pkg;
Pkg.activate(; temp=true);
Pkg.add(["AMDGPU", "MPI", "Oceananigans", "CUDA", "FFTW", "KernelAbstractions", "SeawaterPolynomials", "OffsetArrays", "JLD2", "Adapt", "GPUArraysCore"]);
using Oceananigans;
include(joinpath(pkgdir(Oceananigans, "test", "test_amdgpu.jl")))
'

export HOME="${ORIG_HOME}"
