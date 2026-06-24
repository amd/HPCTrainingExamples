#!/bin/bash

if [ -n "${CRAY_MPICH_VERSION:-}" ]; then
   echo "Skipped - this is MPICH"
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load openmpi

   ompi_info | grep "MPI extensions"
fi
