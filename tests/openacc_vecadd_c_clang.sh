#!/bin/bash

module load clang

cd ${REPO_DIR}/Pragma_Examples/OpenACC/C/vecadd

make
./vecadd
