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

export HSA_XNACK=1

if ! command -v rocprof-sys-instrument >/dev/null 2>&1; then
   echo "rocprof-sys-instrument not found in PATH; Skip"
   exit 0
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_ArrayAssign

cd Ver2

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

export ROCPROFSYS_USE_PROCESS_SAMPLING=false
rocprof-sys-instrument -o GhostExchange.inst -- ./GhostExchange

if [ -n "${CRAY_MPICH_VERSION:-}" ]; then
   MPIRUN=srun
   MPI_RUN_OPTIONS="cpu-bind=verbose,cores"
   # per-resource placement: tasks per socket (closest NUMA equivalent)
   MPI_RESOURCE_MPI4="--ntasks-per-socket=${NUM_PER_RESOURCE_MPI4}"
   MPI_RESOURCE_MPI16="--ntasks-per-socket=${NUM_PER_RESOURCE_MPI16}"
else
   MPIRUN=mpirun
   MPI_RUN_OPTIONS="--mca coll ^hcoll --bind-to core --report-bindings"
   MPI_MAP_BY="--map-by ppr:2:numa"
   # per-resource placement: ranks per NUMA domain
   MPI_RESOURCE_MPI4="--map-by ppr:${NUM_PER_RESOURCE_MPI4}:numa"
   MPI_RESOURCE_MPI16="--map-by ppr:${NUM_PER_RESOURCE_MPI16}:numa"
fi

# Problem sizes are reduced vs. ver1 to stay under the CTest timeout given
# rocprof-sys instrumentation overhead. -x/-y must be non-zero or the
# example divides by zero while computing rank coordinates.
${MPIRUN} -n 4 ${MPI_RESOURCE_MPI4} \
       rocprof-sys-run -- ./GhostExchange.inst \
       -x 2  -y 2  -i 200 -j 200 -h 2 -t -c -I 100
if [[ ${NUM_PER_RESOURCE_MPI16} -le 4 ]]; then
   ${MPIRUN} -n 16 ${MPI_RESOURCE_MPI16} \
          rocprof-sys-run -- ./GhostExchange.inst \
          -x 4  -y 4  -i 400 -j 400 -h 2 -t -c -I 100
fi

ls -Rl rocprofsys-* |grep perfetto
