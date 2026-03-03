#!/bin/bash

# This test checks that the AMDGPU is visible from Julia 

# NOTE: this test assumes Julia has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/julia_setup.sh

export CUR_DIR=$PWD
curl -fsSL https://install.julialang.org | sh -s -- --yes --add-to-path=no -p=${CUR_DIR}/juliaup_install || true
export PATH=$PATH:"${CUR_DIR}/juliaup_install/bin"
juliaup add 1.12
juliaup default 1.12
julia -e 'using Pkg; Pkg.add("AMDGPU")'
julia -e 'using AMDGPU;display(AMDGPU.devices())'
rm -rf $HOME/.julia ${CUR_DIR}/juliaup_install

