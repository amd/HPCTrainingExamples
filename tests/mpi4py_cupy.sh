#!/bin/bash

# Credits: Samuel Antao AMD

# This test checks basic functionalities
# of mpi4py using cupy

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

module load rocm
module load mpi4py
module load cupy

python3 ${REPO_DIR}/Python/mpi4py/mpi4py_cupy.py

