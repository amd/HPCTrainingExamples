#!/bin/bash

module load gcc/13

cd ${REPO_DIR}/Pragma_Examples/OpenACC/C/vecadd

make
./vecadd
