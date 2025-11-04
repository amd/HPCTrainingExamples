#!/bin/bash

module load rocm
module load amdflang-new
if [ "$?" == "1" ]; then
   module load amdclang
fi
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran
cd complex_saxpy
make
./complex_saxpy

make clean
