#!/bin/bash

# This test imports the mpi4py package in Python to test
# if Python MPI  is installed and accessible

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   module load mpi4py/cray-mpich-${CRAY_MPICH_VERSION}
else
   module load mpi4py
endif

python3 -c 'import mpi4py' 2> /dev/null && echo 'Success' || echo 'Failure'
