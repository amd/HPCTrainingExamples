#!/bin/bash

# This test imports the triton package in Python to test
# if Triton is installed and accessible

# NOTE: this test assumes Triton has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh


module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load pytorch

python3 -c 'import triton' 2> /dev/null && echo 'Success' || echo 'Failure'


