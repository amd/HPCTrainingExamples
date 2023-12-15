#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Intro
make saxpy3
./saxpy3

make clean
