#!/bin/bash

# This test installs Julia in $PWD/.julia 

# NOTE: this test uses the instructions located at 
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/julia_setup.sh


export CUR_DIR=$PWD
curl -fsSL https://install.julialang.org | sh -s -- --yes --add-to-path=no -p=${CUR_DIR}/juliaup_install
export PATH=$PATH:"${CUR_DIR}/juliaup_install/bin"
juliaup add 1.12
juliaup default 1.12
juliaup status
rm -rf $HOME/.julia ${CUR_DIR}/juliaup_install

