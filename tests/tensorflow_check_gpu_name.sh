#!/bin/bash

# This test checks that the GPU 
# seen by TensorFlow is an AMD GPU

# NOTE: this test assumes TensorFlow has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/tensorflow_setup.sh

module purge

module load tensorflow

python3 -c 'from tensorflow.python.client import device_lib ; device_lib.list_local_devices()'
