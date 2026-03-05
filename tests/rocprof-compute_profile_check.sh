#!/bin/bash

# This test checks that
# rocprofiler-compute (formerly omniperf) profile runs

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

if command -v rocprof-compute &> /dev/null; then
    echo "rocprof-compute found at: $(which rocprof-compute)"
else
    echo "loading rocprofiler-compute module"
    module load rocprofiler-compute
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/HIP/saxpy
BUILD_DIR=$(mktemp -d build_XXXXXX)
cd ${BUILD_DIR}
cmake ..
make

export HSA_XNACK=1
rocprof-compute profile -n v1 --no-roof -- ./saxpy

cd ..
rm -rf ${BUILD_DIR}
popd
