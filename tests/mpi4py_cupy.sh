#!/bin/bash

# Credits: Samuel Antao AMD

# This test checks basic functionalities
# of mpi4py using cupy

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

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
   # Do NOT load cray-python here: its interpreter (3.12) cannot import the
   # custom cupy/numpy stack, whose C-extensions are built for the system
   # python3 (3.9) -- that mismatch yields "No module named
   # numpy._core._multiarray_umath". Load the custom mpi4py module instead
   # (built for system python3 + cray-mpich), matching the custom cupy module
   # loaded below so both run under one interpreter.
   module load mpi4py
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load mpi4py
fi

module load cupy

python3 ${REPO_DIR}/Python/mpi4py/mpi4py_cupy.py

