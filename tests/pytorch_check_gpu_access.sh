#!/bin/bash

# This test imports the torch package in Python to test 
# if PyTorch is installed and accessible

# NOTE: this test assumes PyTorch has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/training/sources/scripts/pytorch_setup.sh

module purge

module load pytorch

python3 -c 'import torch; print(torch.cuda.is_available())'

