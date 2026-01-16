#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load hipifly

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

mkdir hipifly_test && cd hipifly_test
cp -a ${REPO_DIR}/hipifly/ .
pushd hipifly/vector_add
rm src/hipifly.h

make DFLAGS="-DENABLE_HIP -I${HIPIFLY_PATH} -fPIE"
./vector_add
popd
rm -rf hipifly_test

