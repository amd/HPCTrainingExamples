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
trap 'rm -rf ${BUILD_DIR}' EXIT
cp ${SRC_DIR}/* ${BUILD_DIR}/
cd ${BUILD_DIR}

mkdir build_test && cd build_test

cmake ..
make

export HSA_XNACK=1
# rocprof-compute needs Python >= 3.10; native_tool_finder.py uses PEP 604
# unions. Fail here, naming the reason, rather than later with a SyntaxError
# from inside the tool that names no cause.
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
  echo "ERROR: rocprof-compute needs Python >= 3.10, but python3 is $(python3 -V 2>&1)."
  echo "ERROR: load a newer Python before running this test."
  exit 1
fi
rocprof-compute profile -n v1 --no-roof -- ./saxpy
