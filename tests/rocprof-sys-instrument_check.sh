#!/bin/bash

# This test checks that the rocprof-sys-instrument
# (formerly omnitrace-instrument)
# binary exists and it is able to instrument

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

if command -v rocprof-sys-instrument &> /dev/null; then
    echo "rocprof-sys-instrument found at: $(which rocprof-sys-instrument)"
else
    echo "loading rocprofiler-systems module"
    module load rocprofiler-systems
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/HIP/Stream_Overlap/0-Orig/
BUILD_DIR=$(mktemp -d build_XXXXXX)
cd ${BUILD_DIR}
cmake ../
make -j

rocprof-sys-instrument -o compute_comm_overlap.inst -- compute_comm_overlap

cd ..
rm -rf ${BUILD_DIR}

popd

