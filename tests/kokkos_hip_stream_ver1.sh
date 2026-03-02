#!/bin/bash

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   module switch PrgEnv-cray PrgEnv-amd
   export CXX=${ROCM_PATH}/llvm/bin/amdclang++
else
   module load amdclang
fi
module load kokkos

CLONE_DIR=$(mktemp -d -p . Chapter13_XXXXXX)
trap "rm -rf ${CLONE_DIR}" EXIT
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 ${CLONE_DIR}
pushd ${CLONE_DIR}/Kokkos/StreamTriad/Ver1
sed -i -e 's/80000000/100000/' StreamTriad.cc

rm -rf build
mkdir build && cd build
CXX=hipcc cmake ..
make
./StreamTriad

popd
