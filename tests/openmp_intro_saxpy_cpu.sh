#!/bin/bash

module load amdclang
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Intro
make saxpy_cpu
./saxpy_cpu

make clean
