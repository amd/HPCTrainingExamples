#!/bin/bash

# This test checks that the AMDGPU is visible from Julia 

# NOTE: this test assumes Julia has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/julia_setup.sh

module load julia

julia -e 'using AMDGPU;display(AMDGPU.devices())'

