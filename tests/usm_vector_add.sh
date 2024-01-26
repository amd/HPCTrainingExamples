#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/USM/vector_add_usm
make
make run

make clean
