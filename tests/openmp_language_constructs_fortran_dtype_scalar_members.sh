#!/bin/bash

module load rocm
module load amdflang-new
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran
cd 7_derived_types
make dtype_scalar_members
./dtype_scalar_members

make clean
