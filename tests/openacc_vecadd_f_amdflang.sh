#!/bin/bash

module load aomp

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenACC/Fortran/vecadd

make
./vecadd
