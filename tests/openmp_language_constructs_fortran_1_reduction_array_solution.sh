#!/bin/bash

module load rocm
module load amdflang-new
if [ "$?" == "1" ]; then
   module load amdclang
fi
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/5_reduction_array/1_reduction_array_solution
make
./reduction_array

make clean
