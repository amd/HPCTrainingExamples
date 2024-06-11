#!/bin/bash

module load gcc/13

cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/saxpy

make
./saxpy
make clean
