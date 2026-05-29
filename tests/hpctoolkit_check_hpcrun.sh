#!/bin/bash

# This test checks that hpcrun
# runs without errors

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

module load hpctoolkit
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/HIP/Stream_Overlap/0-Orig/

SRC_DIR=$(pwd)
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT

cd ${BUILD_DIR}

cmake ${SRC_DIR}
make -j

hpcrun -e CPUTIME -e gpu=rocm -t ./compute_comm_overlap 2
ls hpctoolkit-compute_comm_overlap-measurements*
