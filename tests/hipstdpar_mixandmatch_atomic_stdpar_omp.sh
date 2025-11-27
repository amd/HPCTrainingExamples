#!/bin/bash

export HSA_XNACK=1
if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/HIPStdPar/CXX/MixAndMatch/atomic_stdpar_omp

make
./final
make clean

popd
