#!/bin/bash

module load aomp

cd ${REPO_DIR}/Pragma_Examples/OpenACC/Fortran/vecadd

make
./vecadd
