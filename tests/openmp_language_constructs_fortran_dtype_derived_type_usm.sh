#!/bin/bash

module load rocm
module load amdflang-new
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran
cd 6_derived_types
export HSA_XNACK=1
make dtype_derived_type_usm
./dtype_derived_type_usm

make clean
