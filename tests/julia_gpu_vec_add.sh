#!/bin/bash

# This test checks that Julia can add two vectors on the AMD GPU 

# NOTE: this test assumes Julia has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/julia_setup.sh

module load julia

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd $REPO_DIR/Julia/vec_add
julia vec_add.jl
popd
