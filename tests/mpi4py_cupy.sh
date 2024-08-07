#!/bin/bash

# This test checks basic functionalities
# of mpi4py using cupy 

# NOTE: this test assumes openmpi has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/rocm/sources/scripts/openmpi_setup.sh

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

module purge

module load openmpi
module load cupy

python3 ${REPO_DIR}/MLExamples/mpi4py_cupy.py

