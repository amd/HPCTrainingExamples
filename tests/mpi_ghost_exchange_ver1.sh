#!/bin/bash

export HSA_XNACK=1

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

# Detect Cray MPICH so we can pick the right launcher and binding flags.
# Open MPI uses mpirun with --bind-to/--map-by/--report-bindings; Cray MPICH
# does not understand those and is launched through Slurm's srun instead.
is_cray_mpich() {
  if command -v mpichversion >/dev/null 2>&1 && mpichversion 2>/dev/null | grep -qi cray; then
    return 0
  fi
  [[ -n "${CRAY_MPICH_VERSION:-}" || -n "${CRAY_MPICH_DIR:-}" ]]
}

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_ArrayAssign

cd Ver1

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
if is_cray_mpich; then
  echo "Detected Cray MPICH: using srun launcher"
  MPIRUN="srun"
  MPIRUN_OPTIONS="--cpu-bind=verbose,cores"
  # per-resource placement: tasks per socket (closest NUMA equivalent)
  MPI_RESOURCE_MPI4="--ntasks-per-socket=${NUM_PER_RESOURCE_MPI4}"
  MPI_RESOURCE_MPI16="--ntasks-per-socket=${NUM_PER_RESOURCE_MPI16}"
else
  MPIRUN="mpirun"
  MPIRUN_OPTIONS="--bind-to core --report-bindings"
  # per-resource placement: ranks per NUMA domain
  MPI_RESOURCE_MPI4="--map-by ppr:${NUM_PER_RESOURCE_MPI4}:numa"
  MPI_RESOURCE_MPI16="--map-by ppr:${NUM_PER_RESOURCE_MPI16}:numa"
fi

${MPIRUN} -n 4 ${MPIRUN_OPTIONS} ${MPI_RESOURCE_MPI4} ./GhostExchange \
       -x 2  -y 2  -i 2000 -j 2000 -h 2 -t -c -I 1000
if [[ ${NUM_PER_RESOURCE_MPI16} -le 4 ]]; then
   ${MPIRUN} -n 16 ${MPIRUN_OPTIONS} ${MPI_RESOURCE_MPI16} ./GhostExchange \
          -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c -I 1000
fi
