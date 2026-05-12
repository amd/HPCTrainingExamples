#!/bin/bash

# This test checks that
# rocprofiler-compute (formerly omniperf) profile runs

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

module load rocprofiler-compute &> /dev/null

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
SRC_DIR=${REPO_DIR}/HIP/saxpy
BUILD_DIR=$(mktemp -d)
trap 'rm -rf ${BUILD_DIR}' EXIT
cp ${SRC_DIR}/* ${BUILD_DIR}/
cd ${BUILD_DIR}

mkdir build_test && cd build_test

cmake ..
make

export HSA_XNACK=1
rocprof-compute profile -n rooflines_PDF --roof-only  -- ./saxpy
