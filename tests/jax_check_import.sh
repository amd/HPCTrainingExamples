#!/bin/bash

# This test imports the jax package in Python to test
# if JAX is installed and accessible

# NOTE: this test assumes JAX has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/jax_setup.sh


if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load jax

python3 -c 'import jax' 2> /dev/null && echo 'Success' || echo 'Failure'


