#!/bin/bash

# This test checks that the GPU
# seen by PyTorch is an AMD GPU

# NOTE: this test assumes PyTorch has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh

module purge

module load pytorch

python3 -c "import torch; print(f'device name [0]:', torch.cuda.get_device_name(0))"

