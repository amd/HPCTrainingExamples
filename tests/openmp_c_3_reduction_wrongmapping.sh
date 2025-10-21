#!/bin/bash

module load rocm
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/3_reduction/3_reduction_wrongmapping

make
./reduction
make clean
