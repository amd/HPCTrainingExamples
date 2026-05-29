#!/bin/bash

# This test checks that the AMDGPU is visible from Julia 

# NOTE: this test assumes Julia has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/julia_setup.sh

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

export CUR_DIR=$(mktemp -d)
ORIG_HOME="${HOME}"
export HOME="${CUR_DIR}"
export TMPDIR="${CUR_DIR}"
export JULIA_DEPOT_PATH="${CUR_DIR}/julia_depot"
export JULIA_NUM_PRECOMPILE_TASKS=8
curl -fsSL https://install.julialang.org | sh -s -- --yes --add-to-path=no -p=${CUR_DIR}/juliaup_install
export PATH=$PATH:"${CUR_DIR}/juliaup_install/bin"

juliaup add 1.12
juliaup default 1.12
julia -e 'using Pkg; Pkg.add("AMDGPU")'
julia -e 'using AMDGPU;display(AMDGPU.devices())'

export HOME="${ORIG_HOME}"
rm -rf ${CUR_DIR}
