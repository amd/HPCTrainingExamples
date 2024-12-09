#!/bin/bash

module load rocm
module load hipifly

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

mkdir hipifly_test && hipifly_test
cp -a ${REPO_DIR}/hipifly/.
pushd hipifly/vector_add
rm src/hipifly.h

make DFLAGS="-DENABLE_HIP -${HIPIFLY_PATH} -fPIE"
./vector_add
popd
rm -rf hipifly_test
