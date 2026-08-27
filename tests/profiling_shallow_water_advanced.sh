#!/bin/bash
# Build and run one stage of the Profiling-by-example shallow-water advanced track
# on two ranks. The stage directory name is the only argument, e.g.
#   ./profiling_shallow_water_advanced.sh 5_halo_pipeline
#
# Affinity is left to the launcher here. The tutorial binds each rank with
# gpu_bind.sh, which needs an exclusive allocation to bind whole NUMA nodes, and
# this test is only checking that the stage builds and stays correct.

STAGE=$1

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
   echo "rocm module is not loaded"
   echo "loading default rocm module"
   module load rocm
fi
module load openmpi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
SRC_DIR=${REPO_DIR}/Profiling-by-example/shallow-water/advanced/${STAGE}

# The ROCTx ranges in these stages come from the rocprofiler-sdk marker library,
# which older ROCm installations do not ship.
ROCM_PATH=${ROCM_PATH:-$(hipconfig --rocmpath)}
if [ ! -f "${ROCM_PATH}/include/rocprofiler-sdk-roctx/roctx.h" ]; then
   echo "Skip: rocprofiler-sdk-roctx not found under ${ROCM_PATH}"
   exit 0
fi

BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cd ${BUILD_DIR}

cmake ${SRC_DIR}
make
mpirun -n 2 ./shallow_mpi
