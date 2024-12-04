#!/bin/bash

# This test imports the torchvision package in Python
# to tests if it is installed and accessible

# NOTE: this test assumes torchvision has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh

module purge

module load pytorch

python3 -c 'import torchvision' 2> /dev/null && echo 'Success' || echo 'Failure'


