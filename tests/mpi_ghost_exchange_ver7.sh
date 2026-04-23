#!/bin/bash

export HSA_XNACK=1
module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load amdclang openmpi

if ! command -v rocprof-sys-instrument >/dev/null 2>&1; then
   echo "rocprof-sys-instrument not found in PATH; Skip"
   exit 0
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_ArrayAssign

cd Ver7

SRC_DIR=$(pwd)
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cd ${BUILD_DIR}

cmake ${SRC_DIR}

make

NUMCPUS=`lscpu | grep '^CPU(s):' |cut -d':' -f2 | tr -d ' '`
NUM_GPUS=`rocminfo |grep GPU |grep "Device Type" |wc -l`
NUM_PER_RESOURCE_MPI4=`expr 4 / ${NUM_GPUS}`
NUM_PER_RESOURCE_MPI16=`expr 16 / ${NUM_GPUS}`
rocprof-sys-instrument -o GhostExchange.inst -- ./GhostExchange

# Problem sizes are reduced vs. ver1 to stay under the CTest timeout given
# rocprof-sys instrumentation overhead. -x/-y must be non-zero or the
# example divides by zero while computing rank coordinates.
mpirun -n 4 --bind-to core --map-by ppr:${NUM_PER_RESOURCE_MPI4}:numa  --report-bindings \
       rocprof-sys-run -- ./GhostExchange.inst \
       -x 2  -y 2  -i 200 -j 200 -h 2 -t -c -I 100
if [[ ${NUM_PER_RESOUCE_MPI16} -le 4 ]]; then
   mpirun -n 16 --bind-to core --map-by ppr:${NUM_PER_RESOURCE_MPI16}:numa  --report-bindings \
          rocprof-sys-run -- ./GhostExchange.inst \
          -x 4  -y 4  -i 400 -j 400 -h 2 -t -c -I 100
fi

ls -Rl rocprofsys-* |grep perfetto`
