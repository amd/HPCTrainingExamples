#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Intro
make saxpy6
./saxpy6

make clean
