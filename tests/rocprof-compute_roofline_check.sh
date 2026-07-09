#!/bin/bash

# This test checks that
# rocprofiler-compute (formerly omniperf) profile runs

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
SRC_DIR=${REPO_DIR}/HIP/saxpy
BUILD_DIR=$(mktemp -d)
# Clean up the build dir plus any roofline benchmark lock files this user owns.
# rocprof-compute keys the lock only on the GPU UUID under a shared 1777 dir
# (/tmp/rocprof-compute-benchmark), so a leftover lock owned by one user makes
# open(...,"a") fail with EACCES for the next user and roofline gets skipped.
# Owner-scoped -delete is safe under the sticky bit and only removes our own.
trap 'rm -rf ${BUILD_DIR}; \
  find /tmp/rocprof-compute-benchmark -maxdepth 1 -user "$(id -un)" \
       -name "rocprof-compute-benchmark-*.lock" -delete 2>/dev/null' EXIT
cp ${SRC_DIR}/* ${BUILD_DIR}/
cd ${BUILD_DIR}

mkdir build_test && cd build_test

cmake ..
make

export HSA_XNACK=1
rocprof-compute profile -n rooflines_PDF --roof-only  -- ./saxpy
