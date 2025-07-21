#!/bin/bash

# This test checks that Julia can add two vectors on the AMD GPU 

# NOTE: this test assumes Julia has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/julia_setup.sh

export CUR_DIR=$PWD
curl -fsSL https://install.julialang.org | sh -s -- --yes --add-to-path=no -p=${CUR_DIR}/juliaup_install
export PATH=$PATH:"${CUR_DIR}/juliaup_install/bin"
juliaup add 1.12
juliaup default 1.12
julia -e 'using Pkg; Pkg.add("AMDGPU")'
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd $REPO_DIR/Julia/vec_add
julia vec_add.jl
popd
rm -rf $HOME/.julia ${CUR_DIR}/juliaup_install

