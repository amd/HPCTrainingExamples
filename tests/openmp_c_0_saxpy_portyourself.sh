#!/bin/bash

module load rocm
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/1_saxpy/0_saxpy_portyourself

make
./saxpy
make clean
