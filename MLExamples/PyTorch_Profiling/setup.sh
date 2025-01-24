#!/usr/bin/env bash

# Replace the contents of this script with your software setup!

module load rocm
module load pytorch

if [[ `which mpicc | wc -l` -eq 0 ]]; then
   echo " "
   echo " "
   echo "WARNING: could not find MPI in the system, tests involving MPI will fail"
   echo " "
   echo " "
fi
