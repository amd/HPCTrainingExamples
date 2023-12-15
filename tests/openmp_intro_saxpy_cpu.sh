#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Intro
make saxpy_cpu
./saxpy_cpu

make clean
