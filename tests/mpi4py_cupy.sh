#!/bin/bash

# Credits: Samuel Antao AMD

# This test checks basic functionalities
# of mpi4py using cupy

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load mpi4py
module load cupy

python3 ${REPO_DIR}/Python/mpi4py/mpi4py_cupy.py

