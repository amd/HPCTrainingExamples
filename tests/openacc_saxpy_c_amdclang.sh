#!/bin/bash

module load amd-clang

cd ${REPO_DIR}/Pragma_Examples/OpenACC/C/saxpy

make
./saxpy
make clean
