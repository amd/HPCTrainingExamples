#!/bin/bash

module load amdclang
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Intro
make saxpy4
./saxpy4

make clean
