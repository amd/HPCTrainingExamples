#!/bin/bash

# This test checks that JAX
# can see the GPU

# NOTE: this test assumes JAX has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/jax_setup.sh

if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load jax

python3 -c 'import jax; print(jax.devices())'

