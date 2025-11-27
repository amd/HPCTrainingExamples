#!/bin/bash

export HSA_XNACK=1
if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPStdPar/CXX/saxpy_transform

make
export AMD_LOG_LEVEL=3
./saxpy

make clean
