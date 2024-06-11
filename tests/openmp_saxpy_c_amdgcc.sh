#!/bin/bash

module load amd-gcc

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/saxpy

make
./saxpy
make clean
