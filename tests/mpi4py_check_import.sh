#!/bin/bash

# This test imports the mpi4py package in Python to test
# if Python MPI  is installed and accessible

module load mpi4py

python3 -c 'import mpi4py' 2> /dev/null && echo 'Success' || echo 'Failure'
