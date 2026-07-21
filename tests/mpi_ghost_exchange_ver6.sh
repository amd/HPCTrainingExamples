#!/bin/bash

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
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load amdflang-new >& /dev/null
   if [ "$?" == "1" ]; then
      module load amdclang
   fi
   module load openmpi
fi

unset HSA_XNACK

MPI_RUN_OPTIONS="--mca coll ^hcoll"

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_ArrayAssign

cd Ver6

SRC_DIR=$(pwd)
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cd ${BUILD_DIR}

cmake ${SRC_DIR}
make

if [ -n "${CRAY_MPICH_VERSION:-}" ]; then
   MPIRUN=srun
   MPI_RUN_OPTIONS="--cpu-bind=verbose,cores"
   MPI_MAP_BY=""
   export MPICH_GPU_SUPPORT_ENABLED=1
else
   MPIRUN=mpirun
   MPI_RUN_OPTIONS="--mca coll ^hcoll --bind-to core --report-bindings"
   MPI_MAP_BY="--map-by ppr:2:numa"
fi

NUMCPUS=`lscpu | grep '^CPU(s):' |cut -d':' -f2 | tr -d ' '`

if [ ${NUMCPUS} -gt 255 ]; then
   ${MPIRUN} ${MPI_RUN_OPTIONS} -n 16 ${MPI_MAP_BY} ./GhostExchange \
       -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c -I 1000
else
   ${MPIRUN} ${MPI_RUN_OPTIONS} -n 4 ./GhostExchange \
       -x 2  -y 2  -i 2000 -j 2000 -h 2 -t -c -I 1000
fi
