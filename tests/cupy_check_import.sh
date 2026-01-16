#!/bin/bash

# This test imports the cupy package in Python to test
# if Cupy is installed and accessible

# NOTE: this test assumes CuPy has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/cupy_setup.sh


module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load cupy

python3 -c 'import cupy' 2> /dev/null && echo 'Success' || echo 'Failure'


