#!/bin/bash

export HSA_XNACK=1
module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load amdclang openmpi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_ArrayAssign

cd Ver1

rm -rf build
mkdir build && cd build
cmake ..
make

NUMCPUS=`lscpu | grep '^CPU(s):' |cut -d':' -f2 | tr -d ' '`
NUM_GPUS=`rocminfo |grep GPU |grep "Device Type" |wc -l`
NUM_PER_RESOURCE_MPI4=`expr 4 / ${NUM_GPUS}`
NUM_PER_RESOURCE_MPI16=`expr 16 / ${NUM_GPUS}`

mpirun -n 4 --bind-to core --map-by ppr:${NUM_PER_RESOURCE_MPI4}:numa  --report-bindings ./GhostExchange \
       -x 2  -y 2  -i 2000 -j 2000 -h 2 -t -c -I 1000
if [[ ${NUM_PER_RESOUCE_MPI16} -le 4 ]]; then
   mpirun -n 16 --bind-to core --map-by ppr:${NUM_PER_RESOURCE_MPI16}:numa  --report-bindings ./GhostExchange \
          -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c -I 1000
fi

cd ..
rm -rf build
