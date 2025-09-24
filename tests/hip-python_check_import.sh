#!/bin/bash

# This test imports the hip-python package in Python to test
# if HIP-Python is installed and accessible

# NOTE: this test assumes HIP-Python has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/hip-python_setup.sh


module load rocm
module load hip-python

python3 -c 'import hip, hiprtc' 2> /dev/null && echo 'Success' || echo 'Failure'
