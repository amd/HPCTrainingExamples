#!/bin/bash

# This test imports the torchaudio package in Python
# to tests if it is installed and accessible

# NOTE: this test assumes torchaudio has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh



module load rocm pytorch

python3 -c 'import torchaudio' 2> /dev/null && echo 'Success' || echo 'Failure'


