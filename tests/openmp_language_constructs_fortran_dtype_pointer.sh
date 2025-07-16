#!/bin/bash

module load amdflang-new
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/6_derived_types
cd derived_types
make dtype_pointer
./dtype_pointer

make clean
