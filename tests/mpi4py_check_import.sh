#!/bin/bash

# This test imports the mpi4py package in Python to test 
# if Python MPI  is installed and accessible

# NOTE: this test assumes openmpi has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/rocm/sources/scripts/openmpi_setup.sh

module purge

module load openmpi

python3 -c 'import mpi4py' 2> /dev/null && echo 'Success' || echo 'Failure'


