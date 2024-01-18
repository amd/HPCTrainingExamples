#!/bin/bash

export HSA_XNACK=1
module load amdclang openmpi

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

cd ~/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign

cd Ver4

mkdir build && cd build
cmake ..
make

NUMCPUS=`lscpu | grep '^CPU(s):' |cut -d':' -f2 | tr -d ' '`

if [ ${NUMCPUS} -gt 255 ]; then
   mpirun ${MPI_RUN_OPTIONS} -n 16 --bind-to core --map-by ppr:2:numa  --report-bindings ./GhostExchange \
       -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c -I 1000
else
   mpirun ${MPI_RUN_OPTIONS} -n 4 --bind-to core --report-bindings ./GhostExchange \
       -x 2  -y 2  -i 2000 -j 2000 -h 2 -t -c -I 1000
fi

cd ..
rm -rf build
