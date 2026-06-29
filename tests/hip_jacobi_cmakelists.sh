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
   export HIP_PLATFORM=amd
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
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP/jacobi

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   module load libfabric
   MPIRUN=srun
else
   module load openmpi
   MPIRUN=mpirun
fi

SRC_DIR=$(pwd)
# mktemp -d already creates the directory, so do NOT mkdir it again (that
# fails with "File exists", and the && short-circuit then skips the cd,
# leaving us in the source tree). Build out-of-tree and point cmake at the
# absolute source dir -- a relative ".." would resolve under /tmp, not here.
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cd "${BUILD_DIR}"
cmake "${SRC_DIR}"
make

#salloc -p LocalQ --gpus=2 -n 2 -t 00:10:00
${MPIRUN} -n 2 ./Jacobi_hip -g 2
