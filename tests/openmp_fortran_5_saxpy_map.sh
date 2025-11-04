#!/bin/bash

module load rocm
module load amdflang-new >& /dev/null
if [ "$?" == "1" ]; then
   module load amdclang
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/1_saxpy/5_saxpy_map

make
./saxpy
make clean
