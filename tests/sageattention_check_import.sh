#!/bin/bash

# This test imports the sageattention package in Python to test
# if Sage Attention is installed and accessible

# NOTE: this test assumes Sage Attention has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh


module load rocm pytorch

python3 -c 'import sageattention' 2> /dev/null && echo 'Success' || echo 'Failure'


