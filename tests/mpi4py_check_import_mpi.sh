#!/bin/bash

# This test imports the mpi4py package in Python to test
# if Python MPI  is installed and accessible

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
   module load cray-python
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load mpi4py
fi

python3 -c 'from mpi4py import MPI' 2> /dev/null && echo 'Success' || echo 'Failure'


