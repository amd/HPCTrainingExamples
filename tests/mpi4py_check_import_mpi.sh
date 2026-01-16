#!/bin/bash

# This test imports the mpi4py package in Python to test
# if Python MPI  is installed and accessible

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load mpi4py

python3 -c 'from mpi4py import MPI' 2> /dev/null && echo 'Success' || echo 'Failure'


