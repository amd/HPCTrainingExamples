#!/bin/bash

module load rocm

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/hipifly/vector_add

make DFLAGS="-DENABLE_HIP -fPIE"
./vector_add
make clean
