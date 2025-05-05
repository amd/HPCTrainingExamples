#!/bin/bash

module load rocm
XNACK_COUNT=`rocminfo | grep xnack | wc -l`
if [ ${XNACK_COUNT} -lt 1 ]; then
   echo "Skip"
else

   export HSA_XNACK=1
   module load amdclang openmpi omnitrace

   # OpenIB is removed as of OpenMPI 5.0.0, so only needed for older versions
   CurrentVersion=`mpirun --version |head -1 | tr -d '[:alpha:] ) (' `
   RequiredVersion="4.9.9"
   if [ "$(printf '%s\n' "$RequiredVersion" "$CurrentVersion" | sort -Vr | head -n1)" = "$RequiredVersion" ]; then
      echo "Setting MPIRUN options to exclude openib transport layer for mpi version ${CurrentVersion}"
      echo "OpenMPI versions starting with 5.0.0 have the legacy openib transport layer removed"
      MPI_RUN_OPTIONS="--mca pml ob1 --mca btl ^openib"
   else
      MPI_RUN_OPTIONS="--mca coll ^hcoll"
   fi

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_ArrayAssign

   cd Ver2

   rm -rf build
   mkdir build && cd build
   cmake ..
   make

   export OMNITRACE_USE_PROCESS_SAMPLING=false
   omnitrace-instrument -o GhostExchange.inst -- ./GhostExchange
   mpirun -n 4 omnitrace-run -- ./GhostExchange.inst

   ls -Rl omnitrace* |grep perfetto

   cd ..
   rm -rf build
fi
