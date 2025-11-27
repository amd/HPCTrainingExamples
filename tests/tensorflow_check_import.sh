#!/bin/bash

# This test imports the torch package in Python to test
# if TensorFlow is installed and accessible

# NOTE: this test assumes TensorFlow has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/tensorflow_setup.sh


if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load tensorflow

python3 -c 'import tensorflow' 2> /dev/null && echo 'Success' || echo 'Failure'


