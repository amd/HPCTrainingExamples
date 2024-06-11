#!/bin/bash

module load amdclang
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Intro
make saxpy6
./saxpy6

make clean
