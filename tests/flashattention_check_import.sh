#!/bin/bash

# This test imports the flash_attn package in Python to test
# if Flash Attention is installed and accessible

# NOTE: this test assumes Flash Attention has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh


module load rocm pytorch

python3 -c 'import flash_attn' 2> /dev/null && echo 'Success' || echo 'Failure'


