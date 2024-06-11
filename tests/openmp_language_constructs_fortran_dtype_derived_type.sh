#!/bin/bash

module load amdclang
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran
cd derived_types
make dtype_derived_type
./dtype_derived_type

make clean
