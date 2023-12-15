#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Intro
make saxpy1f
./saxpy1f

make clean
