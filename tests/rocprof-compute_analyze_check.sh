#!/bin/bash

# This test checks that
# rocprofiler-compute (formerly omniperf) analyze runs

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

module load rocprofiler-compute &> /dev/null

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/HIP/saxpy
BUILD_DIR=$(mktemp -d build_XXXXXX)
cd ${BUILD_DIR}
cmake ..
make

export HSA_XNACK=1
rocprof-compute profile -n v1 --no-roof -- ./saxpy
rocprof-compute analyze -p workloads/v1/* --block 7.1.0 7.1.1 7.1.2 7.1.0: Grid size 7.1.1: Workgroup size 7.1.2: Total Wavefronts

cd ..
rm -rf ${BUILD_DIR}
popd
