#!/bin/bash

module load amdclang
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/cpp_classes/explicit/daxpy
make
./example
make clean
popd
