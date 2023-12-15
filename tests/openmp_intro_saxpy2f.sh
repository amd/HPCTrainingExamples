#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Intro
make saxpy2f
./saxpy2f

make clean
