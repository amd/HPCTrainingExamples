#!/bin/bash

module load amd-gcc

cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/saxpy

make
./saxpy
make clean
