#!/bin/bash

# This test imports the mpi4py package in Python to test
# if Python MPI  is installed and accessible

module load rocm
module load mpi4py

python3 -c 'from mpi4py import MPI' 2> /dev/null && echo 'Success' || echo 'Failure'


