#!/bin/bash

# This test checks that JAX
# can see the GPU

# NOTE: this test assumes JAX has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/jax_setup.sh

module load rocm
module load jax

python3 -c 'import jax; print(jax.devices())'

