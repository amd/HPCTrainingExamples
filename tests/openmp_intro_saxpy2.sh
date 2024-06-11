#!/bin/bash

module load amdclang
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Intro
make saxpy2
./saxpy2

make clean
