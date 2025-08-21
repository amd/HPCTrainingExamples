#!/bin/bash

module load rocm
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/SingleLineConstructs

export HSA_XNACK=1

make
./saxpy_gpu_collapse
make clean

