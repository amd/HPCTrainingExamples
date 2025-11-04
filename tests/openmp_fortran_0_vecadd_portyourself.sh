#!/bin/bash

module load rocm
module load amdflang-new
if [ "$?" == "1" ]; then
   module load amdclang
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/2_vecadd/0_vecadd_portyourself

make
./vecadd
make clean
