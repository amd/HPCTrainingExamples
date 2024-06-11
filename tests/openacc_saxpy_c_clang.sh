#!/bin/bash

module load clang/15

cd ${REPO_DIR}/Pragma_Examples/OpenACC/C/saxpy

make
./saxpy
make clean
