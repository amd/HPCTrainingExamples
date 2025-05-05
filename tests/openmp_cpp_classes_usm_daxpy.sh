#!/bin/bash

module load amdclang
export HSA_XNACK=1
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/cpp_classes/usm/daxpy
make
./example
make clean
popd
