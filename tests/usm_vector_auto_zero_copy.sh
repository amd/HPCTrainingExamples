#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/USM/vector_add_auto_zero_copy
make
make run

make clean
