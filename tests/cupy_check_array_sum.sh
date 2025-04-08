#!/bin/bash

# This test checks that the copy of arrays 
# from the CPU to the GPU and their sum on the GPU

# NOTE: this test assumes Cupy has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/cupy_setup.sh

module purge

module load cupy

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

pushd $REPO_DIR/MLExamples/ 

python3 cupy_array_sum.py 

module unload cupy

