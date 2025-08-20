#!/bin/bash

export HSA_XNACK=1
module load rocm
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/HIPStdPar/CXX/MixAndMatch/std_cpu_gpu

make
./final
make clean

popd
