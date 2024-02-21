#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran
cd derived_types
make
./dtype_scalar_members

make clean
