#!/bin/bash

# This test imports the cupy package in Python to test 
# if Cupy is installed and accessible

# NOTE: this test assumes Cupy has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/training/sources/scripts/cupy_setup.sh

module purge

module load cupy

python3 -c 'import cupy' 2> /dev/null && echo 'Success' || echo 'Failure'


