#!/bin/bash

# This test imports the torch package in Python to test
# if TensorFlow is installed and accessible

# NOTE: this test assumes TensorFlow has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/tensorflow_setup.sh

module purge

module load tensorflow

python3 -c 'import tensorflow' 2> /dev/null && echo 'Success' || echo 'Failure'


