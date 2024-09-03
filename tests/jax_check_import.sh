#!/bin/bash

# This test imports the jax package in Python to test 
# if JAX is installed and accessible

# NOTE: this test assumes JAX has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/sources/scripts/jax_setup.sh

module purge

module load jax

python3 -c 'import jax' 2> /dev/null && echo 'Success' || echo 'Failure'


