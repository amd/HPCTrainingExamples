#!/bin/bash

module load clacc

cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/saxpy

make
./saxpy
make clean
