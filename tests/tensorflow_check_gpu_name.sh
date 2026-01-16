#!/bin/bash

# This test checks that the GPU
# seen by TensorFlow is an AMD GPU

# NOTE: this test assumes TensorFlow has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/tensorflow_setup.sh


module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load tensorflow

python3 -c 'from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())'
